import datetime
import json
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
from albumentations.pytorch import ToTensorV2
from cjm_pytorch_utils.core import get_torch_device, move_data_to_device, set_seed
from torch import nn
from torch.utils.data import DataLoader
import albumentations as A
import multiprocessing
from tqdm.auto import tqdm
from torch.amp import autocast
from mask_rcnn_train_utils import change_layers_to_eval, initialise_maskrcnn, setup_datasets_training, \
    initialise_maskrcnn_rgbh, setup_training_metadata
from stockpile_dataset_builder import StockpileDatasetBuilderConfig


class TrainConfig:
    # Setup for Training Loop and Initialisation
    NUMBER_CLASSES = StockpileDatasetBuilderConfig.NUM_CLASSES
    NUMBER_WORKERS = multiprocessing.cpu_count() // 2
    DEVICE = torch.device(get_torch_device())
    DTYPE = torch.float32

    # Hyperparameters
    BATCH_SIZE = 4
    LR = 5e-4
    EPOCHS = 500

    # Early Stop
    STOP_AFTER_EPOCHS_WITH_NO_VAL_IMPROVEMENT = 20

    # For Blending
    ALPHA_BLEND = StockpileDatasetBuilderConfig.ALPHA_BLEND

    # Transfer Learning
    TRAINABLE_BACKBONE_LAYERS = 2

    # Parameters for dataset initialisation, data loading, and model loading
    MODE = 'rgb_blend'
    # Careful: D4 rotates images and changes the dimensions. Should only be used with square patches.
    # Also: In 'rgb' ToFloat and ToTensor is expected, see StockpileDataset.setTransforms() for more info.
    TRAIN_AUGS_RGB = A.Compose([
        A.D4(),
        A.ToFloat(max_value=255.0),
        ToTensorV2(transpose_mask=True)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_classes"]))
    # Throws warning because we defined bbox as params but no augmentation that uses bboxes. Warning can be ignored.
    VAL_AUGS_RGB = A.Compose([A.ToFloat(max_value=255.0), ToTensorV2(transpose_mask=True)],
                             bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_classes"]))
    # Careful: RGBH training does normalisation internally, so do not use ToFloat and ToTensor.
    # See StockpileDataset.setTransforms() for more info. No val augs necessary for rgbh training.
    TRAIN_AUGS_RGBH = A.Compose([A.D4()],
                                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_classes"]),
                                additional_targets={'image0': 'image'})

    DATALOADER_PARAMS = {
        'batch_size': BATCH_SIZE,  # Batch size for data loading
        'num_workers': NUMBER_WORKERS,  # Number of subprocesses to use for data loading
        'persistent_workers': True,
        # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once.
        # This allows to maintain the worker dataset instances alive.
        'pin_memory': 'cuda' in DEVICE.type,
        # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using
        # GPU.
        'pin_memory_device': DEVICE.type if 'cuda' in DEVICE.type else '',
        # Specifies the device where the data should be loaded. Commonly set to use the GPU.
        'collate_fn': lambda batch: tuple(zip(*batch)),
    }

    # Paths and Folders
    PROJECT_DIR = "mask_rcnn_models"
    MODEL_DIR = os.path.join(PROJECT_DIR, f"trained_model_{MODE}_1024_1024_{TRAINABLE_BACKBONE_LAYERS}_"
                                          f"resnet_b{BATCH_SIZE}_after_refactor")
    MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "mask_rcnn_resnet50_fpn.pt")


# https://christianjmills.com/posts/pytorch-train-mask-rcnn-tutorial/
# with multiple adjustments
def run_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
              device: torch.device, scaler: torch.GradScaler,
              epoch_id: int, is_training: bool):
    """
       Run one epoch of training or evaluation.

       Args:
           model (nn.Module): The neural network model (Mask R-CNN).
           dataloader (DataLoader): DataLoader providing the data.
           optimizer (torch.optim.Optimizer): Optimizer to update the model parameters.
           lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
           device (torch.device): Device to run the operations on (CPU or GPU).
           scaler (torch.GradScaler): Gradient scaler for mixed precision training.
           epoch_id (int): Current epoch number.
           is_training (bool): Whether the epoch is for training or validation.

       Returns:
           Tuple[float, Dict[str, float]]: Tuple containing the average sum loss
               for the epoch and a dictionary with average losses for classifier,
               box regression, mask, objectness, and RPN box regression.
    """
    model.train()

    epoch_avg_sum_loss = 0
    epoch_avg_classifier_loss = 0
    epoch_avg_box_reg_loss = 0
    epoch_avg_mask_loss = 0
    epoch_avg_objectness_loss = 0
    epoch_avg_rpn_box_reg_loss = 0
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")  # Initialize a progress bar

    # Loop over the data
    for batch_id, (inputs, targets) in enumerate(dataloader):

        # Move inputs and targets to the specified device
        inputs = torch.stack(inputs).to(device)

        # Forward pass with Automatic Mixed Precision (AMP) context manager
        with autocast(torch.device(device).type):
            if is_training:
                losses = model(inputs.to(device), move_data_to_device(targets, device))
            else:
                with torch.no_grad():
                    # manually change layers because implementation returns no losses in eval
                    change_layers_to_eval(model)
                    losses = model(inputs.to(device), move_data_to_device(targets, device))
                    model.train()  # safety

            # compute different losses
            loss = sum([loss for loss in losses.values()])  # Sum up the losses
            epoch_avg_classifier_loss += losses['loss_classifier'].item()
            epoch_avg_box_reg_loss += losses['loss_box_reg'].item()
            epoch_avg_mask_loss += losses['loss_mask'].item()
            epoch_avg_objectness_loss += losses['loss_objectness'].item()
            epoch_avg_rpn_box_reg_loss += losses['loss_rpn_box_reg'].item()

        # If in training mode, backpropagate the error and update the weights
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            # fresh start for next epoch
            optimizer.zero_grad()

        # Update the total loss
        loss_item = loss.item()
        epoch_avg_sum_loss += loss_item

        # Update the progress bar
        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_avg_sum_loss / (batch_id + 1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()

        # If loss is NaN or infinity, stop training
        if is_training:
            stop_training_message = f"Loss is NaN or infinite at epoch {epoch_id}, batch {batch_id}. Stopping training."
            assert not math.isnan(loss_item) and math.isfinite(loss_item), stop_training_message

    # Cleanup and close the progress bar
    progress_bar.close()

    # Return the average loss for this epoch and a dictionary with all other losses
    elements_in_batch = batch_id + 1

    other_losses_dict = {
        'loss_classifier': epoch_avg_classifier_loss / elements_in_batch,
        'loss_box_reg': epoch_avg_box_reg_loss / elements_in_batch,
        'loss_mask': epoch_avg_mask_loss / elements_in_batch,
        'loss_objectness': epoch_avg_objectness_loss / elements_in_batch,
        'loss_rpn_box_reg': epoch_avg_rpn_box_reg_loss / elements_in_batch}

    return epoch_avg_sum_loss / elements_in_batch, other_losses_dict


# https://christianjmills.com/posts/pytorch-train-mask-rcnn-tutorial/
# with multiple adjustments
def train_loop(model: nn.Module,
               train_dataloader: DataLoader,
               valid_dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
               device: torch.device,
               maximum_number_epochs: int,
               checkpoint_path: Path,
               training_parameters: Dict[str, Any],
               stop_after_epochs_with_no_val_improvement: int,
               use_scaler: bool = False):
    """
        Perform the training loop over a specified number of epochs, evaluating on a validation set.

        Args:
            model (nn.Module): The neural network model to be trained.
            train_dataloader (DataLoader): DataLoader providing the training data.
            valid_dataloader (DataLoader): DataLoader providing the validation data.
            optimizer (torch.optim.Optimizer): Optimizer to update the model parameters.
            lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
            device (torch.device): Device to run the operations on (CPU or GPU).
            maximum_number_epochs (int): Maximum number of epochs to train the model
                (in case early stop does not happen).
            checkpoint_path (Path): Path to save the best model checkpoint.
            training_parameters (Dict[str, Any]): Additional parameters related to training.
            stop_after_epochs_with_no_val_improvement (int): Number of epochs to wait before stopping
                if validation loss does not improve.
            use_scaler (bool, optional): Whether to use a gradient scaler for mixed-precision training.
                Defaults to False.

        Returns:
            None
    """
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')  # Initialize the best validation loss
    training_per_epoch = []
    no_improvement_loss = 0

    # Loop over the epochs
    for epoch in tqdm(range(maximum_number_epochs), desc="Epochs"):

        if no_improvement_loss == stop_after_epochs_with_no_val_improvement:
            print(f"Initialized early stopping at epoch {epoch}")
            training_parameters['early_stop_epoch'] = epoch
            break  # early stop

        # Run a training epoch and get the training loss
        train_loss, epoch_all_other_avg_losses_per_epoch_train = run_epoch(model, train_dataloader,
                                                                           optimizer,
                                                                           lr_scheduler,
                                                                           device,
                                                                           scaler, epoch,
                                                                           is_training=True)
        # Run an evaluation epoch and get the validation loss
        with torch.no_grad():
            valid_loss, epoch_all_other_avg_losses_per_epoch_val = run_epoch(model, valid_dataloader,
                                                                             None, None,
                                                                             device, scaler,
                                                                             epoch,
                                                                             is_training=False)

        training_per_epoch.append({
            'epoch': epoch,
            'train_avg_loss': train_loss,  # saved extra because most important
            'valid_avg_loss': valid_loss,  # saved extra because most important
            'epoch_all_other_avg_losses_per_epoch_train': epoch_all_other_avg_losses_per_epoch_train,
            'epoch_all_other_avg_losses_per_epoch_val': epoch_all_other_avg_losses_per_epoch_val,
            'learning_rate': lr_scheduler.get_last_lr()[0],
            'model_architecture': model.name
        })

        # If the validation loss is lower than the best validation loss seen so far, save the model checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            no_improvement_loss = 0
            torch.save(model.state_dict(), checkpoint_path)

            # Save metadata about the best model
            training_metadata_best_model = {
                'epoch': epoch,
                'train_avg_loss': train_loss,
                'valid_avg_loss': valid_loss,
                'epoch_all_other_avg_losses_per_epoch_train': epoch_all_other_avg_losses_per_epoch_train,
                'epoch_all_other_avg_losses_per_epoch_val': epoch_all_other_avg_losses_per_epoch_val,
                'learning_rate': lr_scheduler.get_last_lr()[0],
                'model_architecture': model.name
            }
            with open(Path(checkpoint_path.parent / 'training_metadata_best_model.json'), 'w') as f:
                json.dump(training_metadata_best_model, f)
        else:
            no_improvement_loss += 1

    # Save metadata for all epochs
    with open(Path(checkpoint_path.parent / 'training_metadata_all_losses.json'), 'w') as f:
        json.dump(training_per_epoch, f)

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()


if __name__ == '__main__':
    # Set seed
    seed = 123
    set_seed(seed)
    datasets_have_been_built = False

    if TrainConfig.MODE == 'rgb' or TrainConfig.MODE == 'rgb_blend':
        train_dataset, val_dataset = setup_datasets_training(TrainConfig.MODE,
                                                             augmentations_train=TrainConfig.TRAIN_AUGS_RGB,
                                                             augmentations_val=TrainConfig.VAL_AUGS_RGB,
                                                             datasets_have_been_built=datasets_have_been_built)
        model_mask_rcnn = initialise_maskrcnn(device=TrainConfig.DEVICE, dtype_device=TrainConfig.DTYPE,
                                              trainable_backbone_layers=TrainConfig.TRAINABLE_BACKBONE_LAYERS,
                                              number_of_classes=TrainConfig.NUMBER_CLASSES)
    elif TrainConfig.MODE == 'rgbh':
        train_dataset, val_dataset = setup_datasets_training(TrainConfig.MODE,
                                                             augmentations_train=TrainConfig.TRAIN_AUGS_RGBH,
                                                             datasets_have_been_built=datasets_have_been_built)
        model_mask_rcnn = initialise_maskrcnn_rgbh(device=TrainConfig.DEVICE, dataset=train_dataset,
                                                   dtype_device=TrainConfig.DTYPE,
                                                   trainable_backbone_layers=TrainConfig.TRAINABLE_BACKBONE_LAYERS,
                                                   number_of_classes=TrainConfig.NUMBER_CLASSES)
    else:
        raise Exception("Training mode must be 'rgb', 'rgb_blend' or 'rgbh'")

    # Setup metadata for training
    training_parameters = setup_training_metadata(TrainConfig.MODE, train_dataset.get_image_width(),
                                                  train_dataset.get_image_height(),
                                                  TrainConfig.TRAINABLE_BACKBONE_LAYERS,
                                                  TrainConfig.BATCH_SIZE, TrainConfig.LR,
                                                  TrainConfig.EPOCHS,
                                                  TrainConfig.STOP_AFTER_EPOCHS_WITH_NO_VAL_IMPROVEMENT,
                                                  TrainConfig.NUMBER_CLASSES,
                                                  None if (TrainConfig.MODE == 'rgb' or TrainConfig.MODE == 'rgbh')
                                                  else TrainConfig.ALPHA_BLEND)

    # Initialise Data Loaders
    train_dataloader = DataLoader(train_dataset, **TrainConfig.DATALOADER_PARAMS, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, **TrainConfig.DATALOADER_PARAMS)

    # Initialise optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(model_mask_rcnn.parameters(), lr=TrainConfig.LR)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=TrainConfig.LR,
                                                       total_steps=TrainConfig.EPOCHS * len(train_dataloader))

    # Add optimizer and LR Scheduler to training parameters
    training_parameters.update({'optimizer': type(optimizer).__name__})
    training_parameters.update({'lr_scheduler': type(lr_scheduler).__name__})

    # Create folder and directories
    os.makedirs(TrainConfig.PROJECT_DIR, exist_ok=True)
    os.makedirs(TrainConfig.MODEL_DIR, exist_ok=True)
    checkpoint_path = Path(TrainConfig.MODEL_CHECKPOINT_PATH)

    # Execute Training
    train_loop(model=model_mask_rcnn,
               train_dataloader=train_dataloader,
               valid_dataloader=valid_dataloader,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               device=torch.device(TrainConfig.DEVICE),
               maximum_number_epochs=TrainConfig.EPOCHS,
               checkpoint_path=checkpoint_path,
               training_parameters=training_parameters,
               stop_after_epochs_with_no_val_improvement=TrainConfig.STOP_AFTER_EPOCHS_WITH_NO_VAL_IMPROVEMENT,
               use_scaler=True)

    # Calculate end time and save metadata results
    training_parameters['end_time'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(Path(checkpoint_path.parent / 'training_metadata_parameters'), 'w') as file:
        json.dump(training_parameters, file)

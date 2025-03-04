##########################################################################################################
# Tests are mainly a sanity check because implementations are not based on library.
##########################################################################################################
from typing import List

import pytest
import torch

from mask_rcnn_eval import compute_average_precision_and_average_recall, compute_f1_score, compute_average_box_iou, \
    compute_average_mask_iou


@pytest.mark.parametrize(
    "bboxes_or_masks_gts, bboxes_or_masks_model, mode, iou_threshold, expected_precision, expected_recall", [
        # Empty ground truth and model predictions
        ([], [], 'box', 0.5, 0.0, 0.0),

        # Empty ground truth and non-empty model predictions
        ([torch.empty((0, 4))], [torch.tensor([[0.1, 0.1, 0.2, 0.2]])], 'box', 0.5, 0.0, 0.0),

        ([torch.empty((0, 4)), torch.tensor([[0.3, 0.3, 0.4, 0.4]])], [torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
                                                                       torch.tensor([[0.3, 0.3, 0.4, 0.4]])],
         'box', 0.5, 0.5, 1.0),

        ([torch.empty((0, 1, 1)), torch.tensor([[0, 0, 1, 1]])], [torch.tensor([[1]]),
                                                                  torch.tensor([[0, 0, 1, 1]])],
         'mask', 0.5, 0.5, 1.0),

        # Non-empty ground truth and empty model predictions
        ([torch.tensor([[0.1, 0.1, 0.2, 0.2]])], [torch.empty((0, 4))], 'box', 0.5, 0.0, 0.0),

        ([torch.tensor([[0.1, 0.1, 0.2, 0.2]]), torch.tensor([[0.3, 0.3, 0.4, 0.4]])], [torch.empty((0, 4)),
                                                                                        torch.tensor(
                                                                                            [[0.3, 0.3, 0.4, 0.4]])],
         'box', 0.5, 1.0, 0.5),

        ([torch.tensor([[1]]), torch.tensor([[0, 0, 1, 1]])], [torch.empty((0, 1, 1)),
                                                               torch.tensor([[0, 0, 1, 1]])],
         'mask', 0.5, 1.0, 0.5),

        # Matching single box with 0.5 IoU
        ([torch.tensor([[0.1, 0.1, 0.2, 0.2]])], [torch.tensor([[0.1, 0.1, 0.2, 0.2]])], 'box', 0.5, 1.0, 1.0),

        # Matching with high IoU
        ([torch.tensor([[0.1, 0.1, 0.2, 0.2]])], [torch.tensor([[0.1, 0.1, 0.2, 0.2]])], 'box', 0.99999, 1.0, 1.0),

        # Non-matching single box with 0.5 IoU
        ([torch.tensor([[0.1, 0.1, 0.2, 0.2]])], [torch.tensor([[0.5, 0.5, 0.6, 0.6]])], 'box', 0.5, 0.0, 0.0),

        # Multiple boxes with partial matches
        ([torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])],
         [torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]])], 'box', 0.5, 0.5, 0.5),

        ([torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]])],
         [torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.6, 0.6, 0.7, 0.7]])], 'box', 0.5, 0.5, 0.5),

        # Mask mode with perfect match
        ([torch.tensor([[[1, 1], [1, 1]]], dtype=torch.uint8)], [torch.tensor([[[1, 1], [1, 1]]], dtype=torch.uint8)],
         'mask', 0.5, 1.0, 1.0),

        # Mask mode with partial match
        ([torch.tensor([[[1, 1], [1, 1]]], dtype=torch.uint8)], [torch.tensor([[[1, 0], [0, 1]]], dtype=torch.uint8)],
         'mask', 0.5, 0.0, 0.0),

        # More ground truth boxes than predicted boxes
        ([torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])],
         [torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]])], 'box', 0.5, 1.0, 0.666),

        # More predicted boxes than ground truth boxes
        ([torch.tensor([[0.1, 0.1, 0.2, 0.2]])],
         [torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])], 'box', 0.5, 0.333, 1.0),

        # No overlap between ground truth and any box
        ([torch.tensor([[0.0001, 0.0001, 0.0002, 0.0002]])],
         [torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])], 'box', 0.5, 0, 0),

    ])
def test_compute_average_precision_and_average_recall(bboxes_or_masks_gts, bboxes_or_masks_model, mode, iou_threshold,
                                                      expected_precision, expected_recall):
    precision, recall = compute_average_precision_and_average_recall(bboxes_or_masks_gts, bboxes_or_masks_model, mode,
                                                                     iou_threshold)
    expected_f1_score = (2 * precision * recall) / (precision + recall + 1e-7)
    assert precision == pytest.approx(expected_precision, 0.01)
    assert recall == pytest.approx(expected_recall, 0.01)
    assert compute_f1_score(precision, recall) == pytest.approx(expected_f1_score, 0.01)


@pytest.mark.parametrize("bboxes_gts, bboxes_model, threshold, expected", [
    # Case 1: No ground truth boxes, no model boxes
    ([], [], 0.5, 0),

    # Case 2: Ground truth boxes present, no model boxes
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0]])], [], 0.5, 0),

    # Case 3: No ground truth boxes, model boxes present
    ([], [torch.tensor([[0.0, 0.0, 1.0, 1.0]])], 0.5, 0),

    # Case 4: Single box perfect match
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0]])], [torch.tensor([[0.0, 0.0, 1.0, 1.0]])], 0.5, 1.0),

    # Case 5: Single box no overlap
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0]])], [torch.tensor([[1.0, 1.0, 2.0, 2.0]])], 0.5, 0),

    # Case 6: Multiple boxes with partial overlap
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])],
     [torch.tensor([[0.5, 0.5, 1.5, 1.5], [1.5, 1.5, 2.5, 2.5]])], 0.1, 0.14285714285),

    # Case 7: Multiple boxes with one perfect match
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])],
     [torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])], 0.5, 1.0),

    # Case 8: No match despite being the same because threshold must be EXCEEDED
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0]])], [torch.tensor([[0.0, 0.0, 1.0, 1.0]])], 1.0, 0),

    # Case 9: Threshold lower than all IoUs
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])],
     [torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])], 0.1, 1.0),

    # Case 10: Varying sizes and positions
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])],
     [torch.tensor([[0.0, 0.0, 0.5, 0.5], [2.0, 2.0, 3.0, 3.0]])], 0.249, 0.625),

    # Case 11: Overlapping boxes but below threshold
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0]])], [torch.tensor([[0.5, 0.5, 1.5, 1.5]])], 0.8, 0),

    # Case 12: One image with boxes, one without
    ([torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([])],
     [torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([])], 0.5, 1.0),

    # Case 13: Mixed sizes
    ([torch.tensor([[0.0, 0.0, 0.5, 0.5], [1.0, 1.0, 2.0, 2.0]])],
     [torch.tensor([[0.0, 0.0, 0.5, 0.5], [1.5, 1.5, 2.5, 2.5]])], 0.1, 0.57142857142),

    # Case 14: Large number of boxes
    ([torch.tensor([[i, i, i + 1, i + 1] for i in range(100)])],
     [torch.tensor([[i, i, i + 1, i + 1] for i in range(100)])], 0.5, 1.0),
])
def test_compute_average_box_iou(bboxes_gts: List[torch.Tensor], bboxes_model: List[torch.Tensor], threshold: float,
                                 expected: float):
    result = compute_average_box_iou(bboxes_gts, bboxes_model, threshold)
    assert pytest.approx(result, 0.01) == expected


# Parameterized tests with binary masks
@pytest.mark.parametrize("masks_gts, masks_model, iou_threshold, expected", [
    # Test case 1: Empty masks for both ground truth and model
    ([], [], 0.5, 0.0),

    # Test case 2: Empty ground truth masks
    ([torch.tensor([], dtype=torch.uint8)], [torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)], 0.5, 0.0),

    # Test case 3: Empty model masks
    ([torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)], [torch.tensor([], dtype=torch.uint8)], 0.5, 0.0),

    # Test case 4: Perfect overlap
    (
    [torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)], [torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)], 0.5, 1.0),

    (
    [torch.tensor([[0, 1], [1, 1]], dtype=torch.uint8)], [torch.tensor([[1, 1], [0, 0]], dtype=torch.uint8)], 0.5, 1.0),


    (
    [torch.tensor([[0, 1], [1, 1]], dtype=torch.uint8)], [torch.tensor([[0, 1], [1, 1]], dtype=torch.uint8)], 0.5, 1.0),

    # Test case 7: No intersection or union
    ([torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.uint8)],
     [torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.uint8)], 0.5, 0.0),
])
def test_compute_average_mask_iou(masks_gts: List[torch.Tensor], masks_model: List[torch.Tensor], iou_threshold: float,
                                  expected: float):
    result = compute_average_mask_iou(masks_gts, masks_model, iou_threshold)
    assert result == pytest.approx(expected, rel=1e-2)

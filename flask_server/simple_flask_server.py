from glob import glob
from io import BytesIO
import os
import sys
from zipfile import ZipFile
from flask import Flask, abort, json, redirect, render_template, request, send_file, url_for, Response

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from inference import InferenceConfig, create_convex_hull_polygons, create_merged_predictions, create_model_predictions, \
    create_polygons_with_volume, create_qgis_multipolygons_with_merged_predictions, create_tensor_patches, \
    save_multipolygons_for_leaflet, try_to_align_rasters

# Note: Currently the server is linked to one InferenceConfig (only one request at a time can be handled),
# and reprocessing requires starting from the '/' route. Creating the polygons also happens synchronously.

app = Flask(__name__, static_url_path='/static')

app.config['SERVER_NAME'] = '127.0.0.1:5000'
app.config.update(
    input_directory=InferenceConfig.DIRECTORY_INPUTS,
    orthophoto_path=InferenceConfig.PATH_ORTHOPHOTO,
    dsm_path=InferenceConfig.PATH_DSM,
    dtm_path=InferenceConfig.PATH_DTM,
    upload_folder__polygons=InferenceConfig.DIRECTORY_LEAFLET_POLYGONS
)


@app.route("/", methods=['GET'])
def render_form():
    """
        Renders the form template.

        Returns:
            render_template: HTML template for the settings form.
        """
    try:
        return render_template("form.html")
    except Exception:
        abort(404)


@app.route("/map", methods=['GET'])
def render_map():
    """
        Renders the map template.

        Returns:
            render_template: HTML template for the map.
    """
    try:
        return render_template("map.html")
    except Exception:
        abort(404)


@app.route('/orthophoto', methods=['GET'])
def get_orthophoto():
    """
        Sends the orthophoto file for download.

        Returns:
            send_file: Orthophoto file as an attachment.
    """
    try:
        return send_file(path_or_file=app.config["orthophoto_path"], as_attachment=True)
    except Exception:
        abort(404)


@app.route('/polygons', methods=['GET'])
def get_polygons():
    """
       Creates a zip file containing all GeoJSON polygons and sends it for download.

       Returns:
           send_file: Zip file containing GeoJSON polygons as an attachment.
    """
    try:
        target_directory = app.config["upload_folder__polygons"]

        stream = BytesIO()

        with ZipFile(stream, 'w') as zf:
            for file in glob(os.path.join(target_directory, '*.geojson')):
                zf.write(file, os.path.basename(file))
        stream.seek(0)

        return send_file(stream, as_attachment=True, download_name='polygons.zip')
    except Exception:
        abort(404)


@app.route('/form_parameters', methods=['POST'])
def receive_form_parameters():
    """
       Receives form parameters, initializes configuration, and redirects to the map. Currently,
       the server is linked to one InferenceConfig (only one form at a time can be handled).

       Returns:
           redirect: Redirects to the map page after form submission.
    """
    try:
        min_area_volume = int(request.form.get('min-area-volume'))
        sand_density_volume = float(request.form.get('sand-density-volume').replace(',', '.'))
        InferenceConfig.initialize(min_area_volume, sand_density_volume)

        return redirect(url_for('render_map'))
    except Exception:
        abort(404)


@app.route('/mapWithPolygons', methods=['GET'])
def create_volume_polygons():
    """
        Executes processing steps to create volume polygons and saves them. This method
        creates the polygons synchronously!

        Returns:
            Response: JSON response indicating success or failure.
    """
    try:
        processing_steps = [
            try_to_align_rasters,
            create_tensor_patches,
            create_model_predictions,
            create_merged_predictions,
            create_qgis_multipolygons_with_merged_predictions,
            create_convex_hull_polygons,
            create_polygons_with_volume,
            save_multipolygons_for_leaflet
        ]

        for step in processing_steps:
            step()

        return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}

    except Exception as e:
        abort(Response(str(e)))


if __name__ == '__main__':
    app.run(use_reloader=False, debug=False)

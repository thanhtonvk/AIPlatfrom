from flask import Flask, render_template, request

from modules.classifiy.image_classification.keras.model import structured_dict, layers_dict, losses_dict, \
    optimizes_dict, ModelFromScratch
import zipfile
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/classify', methods=['GET'])
def classify():
    return render_template('classify/index.html')


@app.route('/classify/image', methods=['GET', 'POST'])
def image_classification():
    models = []
    for structured in structured_dict:
        for model in list(structured_dict[structured]['structured'].keys()):
            models.append(model)
    layers = list(layers_dict.keys())
    optimizations = list(optimizes_dict.keys())
    losses = list(losses_dict.keys())
    structs = {'models': models, 'layers': layers, 'optimizes': optimizations, 'losses': losses}
    if request.method == 'POST':
        model = request.form['model']
        optimize = request.form['optimize']
        loss = request.form['loss']
        epochs = request.form['epochs']
        width = request.form['width']
        model_type = request.form['model_type']
        f = request.files['file']
        file_name = f.filename
        file_path = f"/files/{file_name}"
        f.save(file_path)
        folder = f"/files/{file_name.split('.')[0]}"
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        if model == "Scratch":
            base_model = ModelFromScratch(image_size=width, epochs=epochs, optimize=optimizes_dict[optimize],
                                          loss=losses_dict[loss], export_model=model_type)
            base_model.train()
            Accuracy, Precision, Recall, f1_Score, Roc, Specificity, Sensitivity = base_model.evaluate()
            base_model.export()


    return render_template('classify/image_classify.html', structs=structs)


@app.route('/classify/tabular', methods=['GET', 'POST'])
def tabular_classification():
    return render_template('classify/tabular_classify.html')


@app.route('/object-detection', methods=['GET'])
def object_detection():
    return render_template('object_detection/index.html')


@app.route('/regression', methods=['GET'])
def regression():
    return render_template('regression/index.html')


if __name__ == '__main__':
    app.run(host='localhost', port=5000)

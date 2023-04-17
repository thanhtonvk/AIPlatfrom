import zipfile

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/classify', methods=['GET'])
def classify():
    return render_template('classify/index.html')


@app.route('/classify/image', methods=['GET', 'POST'])
def image_classification():
    from modules.classifiy.image_classification.keras.model import structured_dict, layers_dict, losses_dict, \
        optimizes_dict, ModelFromScratch, TransferLearningModel
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
        print(model)
        optimize = request.form['optimize']
        loss = request.form['loss']
        epochs = int(request.form['epochs'])
        width = int(request.form['width'])
        model_type = request.form['model_type']
        num_classes = int(request.form['num-classes'])
        print('post')
        f = request.files['file']
        file_name = f.filename
        file_path = f"./files/{file_name}"
        f.save(file_path)
        folder = f"./files/{file_name.split('.')[0]}"
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        if model == "Scratch":
            base_model = ModelFromScratch(num_classes=num_classes, image_size=width, epochs=epochs,
                                          optimize=optimizes_dict[optimize](),
                                          loss=losses_dict[loss](), export_model=model_type, image_path=folder)
        else:
            base_model = TransferLearningModel(num_classes=num_classes, structured=model, image_size=width,
                                               epochs=epochs,
                                               optimize=optimizes_dict[optimize](),
                                               loss=losses_dict[loss](), export_model=model_type, image_path=folder)
        base_model.train()
        structs['evaluate'] = base_model.evaluate()
        path = base_model.export()
        structs['file_export'] = path

        return render_template('classify/image_classify.html', structs=structs)

    return render_template('classify/image_classify.html', structs=structs)


@app.route('/classify/tabular', methods=['GET', 'POST'])
def tabular_classification():
    from modules.classifiy.tabular_classification.cnn import backbone_model_dict, layers_dict, losses_dict, \
        optimizes_dict
    from modules.classifiy.tabular_classification.ml import model_dict, ML
    model_dict_ml = list(model_dict.keys())
    model_dict_cnn = list(backbone_model_dict.keys())
    layers = list(layers_dict.keys())
    optimizations = list(optimizes_dict.keys())
    losses = list(losses_dict.keys())
    params = {'ml': model_dict_ml, 'cnn': model_dict_cnn, 'layers': layers, 'optimizes': optimizations,
              'losses': losses}
    if request.method == 'POST':
        model = request.form['model']
        print('POST request')
        f = request.files['file']
        file_name = f.filename
        file_path = f"./files/{file_name}"
        f.save(file_path)
        folder = f"./files/{file_name.split('.')[0]}"
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        model = model_dict[model]
        ml = ML(model, folder + '/data.xlsx', folder + '/label.xlsx')
        ml.train()
        params['evaluate'] = ml.evaluate()
        path = ml.export_model()

        params['file_export'] = path
        return render_template('classify/tabular_classify.html', structs=params)
    print('GET request')
    return render_template('classify/tabular_classify.html', structs=params)


@app.route('/object-detection', methods=['GET'])
def object_detection():
    return render_template('object_detection/index.html')


@app.route('/regression', methods=['GET','POST'])
def regression():
    from modules.regression.cnn import backbone_model_dict, layers_dict, losses_dict, \
        optimizes_dict
    from modules.regression.ml import model_dict, ML
    model_dict_ml = list(model_dict.keys())
    model_dict_cnn = list(backbone_model_dict.keys())
    layers = list(layers_dict.keys())
    optimizations = list(optimizes_dict.keys())
    losses = list(losses_dict.keys())
    params = {'ml': model_dict_ml, 'cnn': model_dict_cnn, 'layers': layers, 'optimizes': optimizations,
              'losses': losses}
    if request.method == 'POST':
        model = request.form['model']
        print('POST request')
        f = request.files['file']
        file_name = f.filename
        file_path = f"./files/{file_name}"
        f.save(file_path)
        folder = f"./files/{file_name.split('.')[0]}"
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        model = model_dict[model]
        ml = ML(model, folder + '/data.xlsx', folder + '/label.xlsx')
        ml.train()
        params['evaluate'] = ml.evaluate()
        path = ml.export_model()

        params['file_export'] = path
        return render_template('regression/index.html', structs=params)
    print('GET request')
    return render_template('regression/index.html', structs=params)


if __name__ == '__main__':
    app.run(host='localhost', port=5000)

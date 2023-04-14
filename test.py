from modules.classifiy.image_classification.keras.model import structured_dict
if __name__ == '__main__':
    models = []
    for structured in structured_dict:
        for model in list(structured_dict[structured]['structured'].keys()):
            models.append(model)
    print(models)
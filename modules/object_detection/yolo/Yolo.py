from modules.object_detection.yolo.ultralytics import YOLO
import os
import shutil
import random
import glob

random.seed(42)

ROOT_TRAIN_IMAGES = 'modules/object_detection/yolo/data/train/images'
ROOT_TRAIN_LABELS = 'modules/object_detection/yolo/data/train/labels'
ROOT_TEST_IMAGES = 'modules/object_detection/yolo/data/test/images'
ROOT_TEST_LABELS = 'modules/object_detection/yolo/data/test/labels'


def load_data(format_image):
    root_images = 'modules/object_detection/yolo/data/images/'
    root_labels = 'modules/object_detection/yolo/data/labels/'
    try:
        os.mkdir(ROOT_TRAIN_IMAGES)
        os.mkdir(ROOT_TRAIN_LABELS)
        os.mkdir(ROOT_TEST_IMAGES)
        os.mkdir(ROOT_TEST_LABELS)
    except:
        files = glob.glob(f"{ROOT_TRAIN_IMAGES}/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{ROOT_TRAIN_LABELS}/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{ROOT_TEST_IMAGES}/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{ROOT_TEST_LABELS}/*")
        for f in files:
            os.remove(f)
    list_files = os.listdir(root_labels)
    random.shuffle(list_files)
    list_test = list_files[:int(len(list_files) * 0.2)]
    list_train = list_files[int(len(list_files) * 0.2):]
    for file_name in list_test:
        file_name = file_name.split('.')[0]
        shutil.move(src=root_images + file_name + '.' + format_image,
                    dst=ROOT_TEST_IMAGES + '/' + file_name + '.' + format_image)
        shutil.move(src=root_labels + file_name + '.txt', dst=ROOT_TEST_LABELS + '/' + file_name + '.txt')

    for file_name in list_train:
        file_name = file_name.split('.')[0]
        shutil.move(src=root_images + file_name + '.' + format_image,
                    dst=ROOT_TRAIN_IMAGES + '/' + file_name + '.' + format_image)
        shutil.move(src=root_labels + file_name + '.txt', dst=ROOT_TRAIN_LABELS + '/' + file_name + '.txt')


def prepare_custom(labels):
    if os.path.exists('modules/object_detection/yolo/datasets/custom_data.yaml'):
        os.remove('modules/object_detection/yolo/datasets/custom_data.yaml')
    file = open('modules/object_detection/yolo/datasets/custom_data.yaml', mode='a')
    content = f"train : {ROOT_TRAIN_IMAGES}/\nval : {ROOT_TEST_IMAGES}/\nis_coco: False\nnc: {len(labels)}\nnames: {labels}"
    file.write(content)
    file.close()


class Yolo:
    def __init__(self, labels, extension_image, version=8, pretrained='n', epochs=100):
        self.version = version
        self.pretrained = pretrained
        self.epochs = epochs
        self.model = YOLO(f"yolov{version}{pretrained}.yaml")
        self.labels = labels
        self.extension_image = extension_image

    def train(self):
        prepare_custom(labels=self.labels)
        load_data(self.extension_image)
        self.model.train(data='/AIPlatform/modules/object_detection/yolo/datasets/custom_data.yaml', epochs=self.epochs)

    def eval(self):
        metrics = self.model.val()
        return metrics

    def export(self, type):
        self.model.export(format=type)


if __name__ == '__main__':
    yolo = Yolo(labels=['card'], extension_image='jpeg', version=5, pretrained='n', epochs=1)
    yolo.train()
    yolo.eval()
    yolo.export(type='tflite')

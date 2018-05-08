import json
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image


def process_data_annotations(dir,filepath):
    labelfile = open(filepath)
    annotation_list = json.load(labelfile)['annotations']
    num_sample = len(annotation_list)
    image_ids = [annotation_list[i]["image_id"] for i in range(num_sample)]
    image_labels = [annotation_list[i]['label_id'] for i in range(num_sample)]

    #过滤掉不存在图片的id及其label
    image_ids_exit=[]
    image_labels_exit = []
    image_path_exit = []
    for i in range(num_sample):
        img_path = '%s/%s.jpg' % (dir, image_ids[i])
        if os.path.exists(img_path):
            image_ids_exit.append(image_ids[i])
            image_labels_exit.append(image_labels[i])
            image_path_exit.append(img_path)

    return image_ids_exit,image_labels_exit,image_path_exit




def data_gen(img_paths,img_labels,batch_size=32,is_shuffle=True):
    num_sample = len(img_paths)
    img_labels = LabelBinarizer().fit_transform(img_labels)
    while True:

        if is_shuffle:
            img_paths,img_labels = shuffle(img_paths, img_labels)

        for offset in range(0,num_sample,batch_size):

            if is_shuffle:
                img_paths, img_labels = shuffle(img_paths, img_labels)

            features = []
            labels = []
            batch_path = img_paths[offset:offset+batch_size]
            batch_labels = img_labels[offset:offset+batch_size]
            for i in range(len(batch_path)):
                img_path = batch_path[i]
                if os.path.exists(img_path):
                    img = image.load_img(img_path, target_size=(224, 224))
                    img = image.img_to_array(img)
                    img = preprocess_input(img)
                    features.append(preprocess_input(img))
                    labels.append(batch_labels[i])
            yield np.array(features), np.array(labels)

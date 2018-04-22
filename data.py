import json
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image

# file = open('f:/fourniture_classification/validation.json')
# train = json.load(file)
#
# print(len(train['annotations']))

def data_gen(dir,labelfile_path,batch_size=32,is_shuffle=True):
    labelfile = open(labelfile_path)
    annotation_list = json.load(labelfile)['annotations']
    num_sample = len(annotation_list)
    image_ids = [annotation_list[i]["image_id"] for i in range(num_sample)]
    image_labels = [annotation_list[i]['label_id'] for i in range(num_sample)]
    image_labels = LabelBinarizer().fit_transform(image_labels)
    while True:

        if is_shuffle:
            image_ids,image_labels = shuffle(image_ids, image_labels)

        for offset in range(0,num_sample,batch_size):

            if is_shuffle:
                image_ids,image_labels = shuffle(image_ids,image_labels)

            features = []
            labels = []
            batch_ids = image_ids[offset:offset+batch_size]
            batch_labels = image_labels[offset:offset+batch_size]
            for i in range(len(batch_ids)):
                img_path = '%s/%s.jpg'%(dir,batch_ids[i])
                if os.path.exists(img_path):
                    img = image.load_img(img_path, target_size=(224, 224))
                    img = image.img_to_array(img)
                    img = preprocess_input(img)
                    # img = cv2.imread(img_path)
                    # img = cv2.resize(img,(224,224))
                    features.append(preprocess_input(img))
                    labels.append(batch_labels[i])
            yield np.array(features), np.array(labels)




# load_data("haha","./README.md")
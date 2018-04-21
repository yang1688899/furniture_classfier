import json
import cv2
import os
import numpy as np

file = open('G:/fourniture_classification/validation.json')
validation = json.load(file)

print(validation['annotations'])

def data_gen(dir,labelfile_path,batch_size=32):
    labelfile = open(labelfile_path)
    annotation_list = json.load(labelfile)['annotations']
    for offset in range(0,len(annotation_list),batch_size):
        features = []
        labels = []
        for i in range(offset,offset+batch_size):
            img_path = '%s/%s.jpg'%(dir,annotation_list[i]["image_id"])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img,(224,224))
                features.append(img)
                labels.append(annotation_list[i]["label_id"])
        yield np.array(features),np.array(labels)




# load_data("haha","./README.md")
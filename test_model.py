from keras.models import load_model
import data
import json
import os
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
from math import ceil
import random
import csv

def process_annotation_test(img_dir,filepath):
    file = open(filepath)
    test_dict = json.load(file)['images']
    print(len(test_dict))
    img_id_all = [test_dict[i]['image_id'] for i in range(len(test_dict))]
    img_paths = []
    img_id_exits = []
    img_id_nonexits = []
    for i in range(len(img_id_all)):
        img_path = '%s/%s.jpg'%(img_dir,img_id_all[i])
        if os.path.exists(img_path):
            img_paths.append(img_path)
            img_id_exits.append(img_id_all[i])
        else:
            img_id_nonexits.append(img_id_all[i])
    return img_id_all, img_id_exits, img_id_nonexits, img_paths

def test_generator(img_paths, batch_size=32):
    num_test = len(img_paths)
    for offset in range(0, num_test, batch_size):
        samples = []
        batch_test = img_paths[offset:offset+batch_size]
        for i in range(len(batch_test)):
            img = image.load_img(batch_test[i], target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(img)
            samples.append(img)
        yield np.array(samples)

def test(model_file,batch_size=32):
    img_id_all, img_id_exits, img_id_nonexits, img_paths = process_annotation_test('G:/fourniture_classification/test',
                                                                 'G:/fourniture_classification/test.json')
    test_gen = test_generator(img_paths,batch_size=batch_size)
    model = load_model(model_file)
    predictions = model.predict_generator(test_gen, steps=ceil(len(img_paths)/batch_size) )
    max_preds = np.argmax(predictions, axis=1)
    #label starts with 1
    max_preds = max_preds+1

    results_exits = [(img_id_exits[i],max_preds[i]) for i in range(len(max_preds))]
    results_nonexits = [(img_id_nonexits[i], int(random.random()*128)) for i in range(len(img_id_nonexits))]

    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id","predicted"])
        writer.writerows(results_exits)
        writer.writerows(results_nonexits)




img_id_all, img_id_exits, img_id_nonexits, img_paths = process_annotation_test('G:/fourniture_classification/test','G:/fourniture_classification/test.json')

print(len(img_id_all),len(img_id_exits),len(img_id_nonexits),len(img_paths))

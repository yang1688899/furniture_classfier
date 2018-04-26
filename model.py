from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten,GlobalAveragePooling2D
import numpy as np
import data
from math import ceil
from keras.models import load_model

num_train = 194828
num_validation = 6400


def network():
    base_model = VGG19(weights='imagenet', include_top=False,input_shape=(224,224,3))
    x = base_model.output

    # x = Flatten()(x)
    # # x = GlobalAveragePooling2D()(x)
    # x = Dense(4096, activation='relu')(x)
    # x = Dense(4096, activation='relu')(x)
    # predictions = Dense(128, activation='softmax')(x)

    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(128, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return  base_model,model

def get_accuracy(batch_size=32):
    # np.max(predictions,axis=0)
    valid_ids, valid_labels, valid_paths = data.process_data_annotations('f:/fourniture_classification/validation',
                                                                         'f:/fourniture_classification/validation.json')
    valid_gen = data.data_gen(valid_paths, valid_labels,
                              batch_size=batch_size, is_shuffle=False)

    model = load_model("./model.h5")
    predictions = model.predict_generator(valid_gen,steps=ceil(len(valid_labels)/batch_size))
    valid_labels = np.array(valid_labels,dtype='int64')

    print(len(predictions))
    print(len(valid_labels))
    print(predictions.shape)
    print(predictions[0])
    print(np.sum(predictions[0]))
    print(np.max(predictions[0]))

    max_preds = np.argmax(predictions, axis=1)
    print(max_preds.shape)
    print(valid_labels.shape)

    print(max_preds)
    print(valid_labels)

    print(type(max_preds[0]))
    print(type(valid_labels[0]))

    acccracy = np.sum([valid_labels == max_preds])/len(predictions)

    return acccracy


def train(batch_size=32):
    # train_gen = data.data_gen('G:/fourniture_classification/validation', 'G:/fourniture_classification/validation.json',batch_size=batch_size)
    # valid_gen = data.data_gen('G:/fourniture_classification/validation', 'G:/fourniture_classification/validation.json',batch_size=batch_size,is_shuffle=False)

    train_ids,train_labels,train_paths = data.process_data_annotations('f:/fourniture_classification/train', 'f:/fourniture_classification/train.json')
    valid_ids,valid_labels,valid_paths = data.process_data_annotations('f:/fourniture_classification/validation', 'f:/fourniture_classification/validation.json')

    train_gen = data.data_gen(train_paths, train_labels,
                              batch_size=batch_size)
    valid_gen = data.data_gen(valid_paths, valid_labels,
                              batch_size=batch_size, is_shuffle=False)
    base_model,model = network()
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    print("begin training......")
    print("train samples: %s"%len(train_labels))
    print("valid samples: %s"%len(valid_labels))

    model.fit_generator(train_gen, steps_per_epoch=ceil(num_train/batch_size), epochs=1,validation_data=valid_gen, validation_steps=ceil(num_validation/batch_size),)

    model.save('./model.h5')
    print("model saved")

# train(batch_size=16)

# model = load_model('./model.h5')
# model.summary()

# print(get_accuracy())

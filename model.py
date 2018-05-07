from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten,GlobalAveragePooling2D,Dropout
import numpy as np
import data
from math import ceil
from keras.models import load_model
from keras import optimizers
import os


def network():
    base_model = VGG19(weights='imagenet', include_top=False,input_shape=(224,224,3))
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(128, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return  base_model,model

def get_accuracy(modelfile,batch_size=32):
    # np.max(predictions,axis=0)
    valid_ids, valid_labels, valid_paths = data.process_data_annotations('f:/fourniture_classification/validation',
                                                                         'f:/fourniture_classification/validation.json')
    valid_gen = data.data_gen(valid_paths, valid_labels,
                              batch_size=batch_size, is_shuffle=False)

    model = load_model(modelfile)
    predictions = model.predict_generator(valid_gen,steps=ceil(len(valid_labels)/batch_size))
    valid_labels = np.array(valid_labels,dtype='int64')

    max_preds = np.argmax(predictions, axis=1)
    acccracy = np.sum([valid_labels == max_preds+1])/len(predictions)

    print('valudation accuracy is: %s'%acccracy)

    loss = model.evaluate_generator(valid_gen,steps=ceil(len(valid_labels)/batch_size))

    print('valudation loss is: %s'%loss)


def train(model_path,save_path,rate=0.00003,epochs=1,batch_size=32,is_full_train=True):
    # train_gen = data.data_gen('G:/fourniture_classification/validation', 'G:/fourniture_classification/validation.json',batch_size=batch_size)
    # valid_gen = data.data_gen('G:/fourniture_classification/validation', 'G:/fourniture_classification/validation.json',batch_size=batch_size,is_shuffle=False)

    train_ids,train_labels,train_paths = data.process_data_annotations('f:/fourniture_classification/train', 'f:/fourniture_classification/train.json')
    valid_ids,valid_labels,valid_paths = data.process_data_annotations('f:/fourniture_classification/validation', 'f:/fourniture_classification/validation.json')

    num_train = len(train_paths)
    num_validation = len(valid_paths)

    train_gen = data.data_gen(train_paths, train_labels,
                              batch_size=batch_size)
    valid_gen = data.data_gen(valid_paths, valid_labels,
                              batch_size=batch_size, is_shuffle=False)

    if os.path.exists(model_path):

        print("loading model from %s"%model_path)
        model = load_model(model_path)
        if is_full_train:
            for layer in model.layers:
                layer.trainable = True
        else:
            for layer in model.layers[:22]:
                layer.trainable = False

        model.summary()
        print("begin training......")
        print("train samples: %s" % len(train_labels))
        print("valid samples: %s" % len(valid_labels))

        adam = optimizers.Adam(lr=rate)
        model.compile(optimizer=adam, loss='categorical_crossentropy')

        model.fit_generator(train_gen, steps_per_epoch=ceil(num_train / batch_size), epochs=epochs,
                            validation_data=valid_gen, validation_steps=ceil(num_validation / batch_size))

        model.save(save_path)
        print("model saved")

    else:
        base_model,model = network()
        for layer in base_model.layers:
            layer.trainable = False

        print("begin training......")
        print("train samples: %s"%len(train_labels))
        print("valid samples: %s"%len(valid_labels))

        sgd = optimizers.SGD(lr=rate)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        model.fit_generator(train_gen, steps_per_epoch=ceil(num_train/batch_size), epochs=epochs,validation_data=valid_gen, validation_steps=ceil(num_validation/batch_size))

        model.save(save_path)

        print("model saved")

def retrain_with_achitature(modelfile,save_path,rate=0.00003,epochs=1,batch_size=32):
    train_ids, train_labels, train_paths = data.process_data_annotations('f:/fourniture_classification/train',
                                                                         'f:/fourniture_classification/train.json')
    valid_ids, valid_labels, valid_paths = data.process_data_annotations('f:/fourniture_classification/validation',
                                                                         'f:/fourniture_classification/validation.json')

    num_train = len(train_paths)
    num_validation = len(valid_paths)

    train_gen = data.data_gen(train_paths, train_labels,
                              batch_size=batch_size)
    valid_gen = data.data_gen(valid_paths, valid_labels,
                              batch_size=batch_size, is_shuffle=False)

    old_model = load_model(modelfile)
    for i, layer in enumerate(old_model.layers):
        print(i, layer.name)
    uper_model = old_model.get_layer('flatten_1')
    x = uper_model.output
    x = Dropout(0.5)(x)
    x = Dense(2042, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(128, activation='softmax')(x)

    model = Model(old_model.input,outputs=predictions)
    for layer in old_model.layers:
        layer.trainable = False
    model.summary()
    print("begin training......")
    print("train samples: %s" % len(train_labels))
    print("valid samples: %s" % len(valid_labels))

    sgd = optimizers.SGD(lr=rate)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    model.fit_generator(train_gen, steps_per_epoch=ceil(num_train / batch_size), epochs=epochs,
                        validation_data=valid_gen, validation_steps=ceil(num_validation / batch_size))

    model.save(save_path)
    print("model saved")

# model = load_model('./model/model_dropout_epoch3.h5')
# model.summary()
#
get_accuracy('./model/model_4fclayer_epoch7.h5')
# train('./model/model_4fclayer_epoch6.h5','./model/model_4fclayer_epoch7.h5',epochs=1,rate=1e-9,batch_size=16,is_full_train=True)
# retrain_with_achitature('./model/model_dropout_epoch5.h5','./model/model_4fclayer_epoch1.h5',rate=0.0003,epochs=1,batch_size=32)
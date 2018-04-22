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
    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    predictions = Dense(128, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return  base_model,model

# def get_accuracy(prediction):
#     np.max

def train(batch_size=32):
    # train_gen = data.data_gen('G:/fourniture_classification/validation', 'G:/fourniture_classification/validation.json',batch_size=batch_size)
    # valid_gen = data.data_gen('G:/fourniture_classification/validation', 'G:/fourniture_classification/validation.json',batch_size=batch_size,is_shuffle=False)
    train_gen = data.data_gen('f:/fourniture_classification/validation', 'f:/fourniture_classification/validation.json',
                              batch_size=batch_size)
    valid_gen = data.data_gen('f:/fourniture_classification/validation', 'f:/fourniture_classification/validation.json',
                              batch_size=batch_size, is_shuffle=False)
    base_model,model = network()
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='Adam', loss='categorical_crossentropy')

    model.fit_generator(train_gen, steps_per_epoch=ceil(num_train/batch_size), epochs=1,validation_data=valid_gen, validation_steps=ceil(num_validation/batch_size))

    model.save('./model.h5')

# train(batch_size=24)

model = load_model('./model.h5')
model.summary()
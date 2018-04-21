from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
import numpy as np
import data

# img = image.load_img('G:/fourniture_classification/validation/1.jpg', target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

def network():
    base_model = VGG19(weights='imagenet', include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    predictions = Dense(80, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return  base_model,model

def train():
    gen = data.data_gen('G:/fourniture_classification/validation', 'G:/fourniture_classification/validation.json')
    base_model,model = network()
    model.summary()
    # for layer in base_model.layers:
    #     layer.trainable = False
    #
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    #
    # model.model.fit_generator()

train()
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D,Input,Flatten,Dense,Layer



def make_embedding():
    inp = Input(shape=(105,105,1), name = "input_image")
    # layer 1
    c1 = Conv2D(64,(10,10),activation="relu")(inp)
    m1 = MaxPool2D(64,(2,2),padding="same")(c1)

    # Layer 2
    c2 = Conv2D(128,(7,7),activation="relu")(m1)
    m2 = MaxPool2D(64,(2,2),padding="same")(c2)

    # Layer 3
    c3 = Conv2D(128,(4,4),activation="relu")(m2)
    m3 = MaxPool2D(64,(2,2),padding="same")(c3)

    # Layer 4
    c4 = Conv2D(256,(4,4),activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096,activation="sigmoid")(f1)

    return Model(inputs = [inp], outputs = [d1],name = "embedding")

embed = make_embedding()

class L1Distace(Layer):
    def __init__(self,**kwargs):
        super().__init__()
    
    def call(self,input_embedding,validation_embedding):
        return tf.math.abs(input_embedding-validation_embedding)


def make_siamese():

    input_image = Input((105,105,1),name="Input_Image")
    validation_image = Input((105,105,1),name = "Validation_image")

    siamense = L1Distace()
    input_embedding = embed(input_image)
    validation_embedding = embed(validation_image)
    dist = siamense.call(input_embedding,validation_embedding)

    #Last Layer

    classify = Dense(1,activation="sigmoid")(dist)

    return Model(inputs = [input_image,validation_image],outputs = [classify], name = "siamense_model")

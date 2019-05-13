# -*- coding:utf-8 -*-
"""
Based on CNN neural network, training handwritten numeral recognition model model.h5.
The class refers to the trained model for handwritten numeral recognition
"""
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import load_model
from GUI.ImageUtils import changeImage28
import os

class Recognition:
    def __init__(self,path):
        K.set_image_dim_ordering('th')
        self.my_model = load_model(path+'/model/model.h5')

    def readImage2result(self, rimg):
        width, height = rimg.size
        im = rimg.convert("L")
        data = im.getdata()
        data = np.matrix(data, dtype='float') / 255.0
        new_data = np.reshape(data * 255.0, (height, width))
        new_data = new_data[np.newaxis, np.newaxis, :]
        pred = self.my_model.predict(new_data)
        return np.argmax([pred[0]])


if __name__ == '__main__':
    base_dir = os.path.dirname(os.getcwd())
    re = Recognition(base_dir)
    rimg = Image.open(base_dir+'/GUI/only.png')
    rimg = changeImage28(rimg)
    result = re.readImage2result(rimg)
    print(result)

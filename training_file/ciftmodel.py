
# limitations under the License.
# ==============================================================================

# Build model of image classification (2 category), with preparing data,
# version: TF 1.10.0   TF-gpu 1.1.0   keras 2.1.5(not use)   Python 3.6.8  flask 1.0.2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import Callback, EarlyStopping
#import matplotlib.pyplot as plt

import numpy as np
import PIL
from PIL import Image
import json
import time
import requests
# from StringIO import StringIO     # python2 version
from io import BytesIO     # python3 version
import os.path
import sys

def loadImageUrl(imgUrl):
    response = requests.get(imgUrl)
    return Image.open(BytesIO(response.content))
def loadImageFile(imgFile):
    return Image.open(imgFile)

def loadImage(imageFile,size=224):
    if imageFile.startswith('http'):
        img = loadImageUrl(imageFile)
    elif os.path.isfile(imageFile):
        img = loadImageFile(imageFile)
    else:
        return {"file": imageFile, "message": "Not Found"}
    img = img.resize((size, size), PIL.Image.ANTIALIAS)
    img = img.convert('RGB')
    img =np.array(img, dtype=int)
    return img

def data_prepared_json(jsondata):

    if isinstance(jsondata, str):
        load_data = json.loads(jsondata)   # if load_data is a string
    else:
        load_data = jsondata               # if load_data is a json format

    dict_data = load_data['training_data'] # this is the 'image_url', 'label' format
    try:
        count = len(dict_data['label'])
    except TypeError:
        count = len(dict_data)
    print('there are',count,'data in all')
    extract_data=np.zeros((count, 224, 224, 3), float)
    extract_labels = np.zeros((count,1),int)
    if count == 1:
        imgUrl = dict_data['image_url']
        img = loadImage(imgUrl)
        extract_data[0] = img
        extract_labels[0] = int(dict_data['label'])
    else:
        for i in range(0, count):
            imgUrl = dict_data[i]['image_url']
            img = loadImage(imgUrl)
            if type(img) is dict:
                print('can not extract image')
                return 0, 0
            else:
                extract_data[i] = img
                extract_labels[i] = int(dict_data[i]['label'])
    return extract_data, extract_labels


# def data_prepared_local(url1,url2):
# # gether all trained and test data
# # url1 is the location of class 1
# # url2 is the location of class 2
# # in test period-1, we add data from local, used pointer to point the image in folder
# # in test period-2, we add data from website,
#
#
# # scenario1: training data from local -- different class of image in different directory,
# #            grad images from these ditectories and convert into data and labels
#     class1_dir = [url1+i for i in os.listdir(url1)]
#     class2_dir = [url2+i for i in os.listdir(url2)]
#     len1 = len(class1_dir)
#     len2 = len(class2_dir)
#     count = len1+len2
#     data = np.zeros((count,224,224,3),float)
#     labels = np.zeros((count,1),int)
#     for i, image_file in enumerate(class1_dir):
#         image = loadImage(image_file)
#         data[i] = image
#         labels[i]=0
#         if i%100 == 0: print('Processing Row Data {} of {}'.format(i, count))
#     for i, image_file in enumerate(class2_dir):
#         image = loadImage(image_file)
#         data[len1+i] = image
#         labels[len1+i]=1
#         if i%100 == 0: print('Processing Row Data {} of {}'.format(i+len1, count))
#     print('finished processing Row Data')
# # suffle data
#     idx=np.random.permutation(count)
#     data_random= data[idx]
#     labels_random = labels[idx]
#     return data_random , labels_random
# # scenario2: training data from web link,


def built_model(num_class=1):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same',  input_shape = (224, 224, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_class, activation='sigmoid'))
    model.summary()

    opt = tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
    if num_class == 1:
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

    def show_lost(self):
        loss = self.losses
        val_loss = self.val_losses

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('my model Loss Trend')
        plt.plot(loss, 'blue', label='Training Loss')
        plt.plot(val_loss, 'green', label='Validation Loss')
        plt.xticks(range(0, 3)[0::2])
        plt.legend()
        plt.show()


def train(model,train_data,train_labels,nb_epoch=2,batch_size=2,split=0.25,early_stop=True):
    history=LossHistory()
    if early_stop == True:
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=nb_epoch,
                  validation_split=split, verbose=1, shuffle=True, callbacks=[history, early_stopping])

    else:
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=nb_epoch,
                  validation_split=split, verbose=1, shuffle=True, callbacks=[history])

    return model, history


def predict(model,imgFile):
    if type(imgFile) is str:
        img = loadImage(imgFile)
    if type(imgFile) is np.ndarray:
        img = imgFile
    img = np.reshape(np.array(img), [1, 224, 224, 3])
    output = model.predict(img,verbose=0)[0]
# show prediction result
    if output>0.5:
        classnum=1
        name = 'Husky'
    else:
        classnum=0
        output=1-output
        name = 'Shepherd'
    print('this image was predicted as %s ，with %f confidence'%(name,output))
    return output

def save_model(model,name):
    model.save('./models/'+name+'.h5')

def load_model(name):
    loaded_model = tf.keras.models.load_model('./models/'+name+'.h5')
    return loaded_model

def show_lost(history):
    loss = history.losses
    val_loss = history.val_losses
    print('training lost is: ', loss, '   validation lost is: ', val_loss)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('my model Loss Trend')
    # plt.plot(loss, 'blue', label='Training Loss')
    # plt.plot(val_loss, 'green', label='Validation Loss')
    # plt.xticks(range(0,3)[0::2])
    # plt.legend()
    # plt.show()

# def show_image(imgURL):
#     if type(imgURL) is str:
#         img=loadImage(imgURL)
#     if type(imgURL) is np.ndarray:
#         img=np.array(imgURL/255)
#     plt.imshow(img)
#     plt.show()


if __name__ == '__main__':
    url1 = './data/1/'
    url2 = './data/2/'
    list_url1=[url1+i for i in os.listdir(url1)]
    list_url2=[url2+i for i in os.listdir(url2)]
    web_url='https://pic2.zhimg.com/80/7405939b62a73f28846533de08db3a80_hd.jpg'
    dog_data, dog_labels = data_prepared_local(url1, url2)

# training precedure
    #cli = built_model()
    #cli,history=train(cli,dog_data,dog_labels,nb_epoch=5,batch_size=8,split=0.3)
    #show_lost(history)
    #history.show_lost()

# load and test procedure
    test_model=load_model('./models/thirdmodel.h5')
    predict(test_model,web_url)

# continue training
#     cli=load_model('secondmodel.h5')
    cli, history = train(test_model, dog_data, dog_labels, nb_epoch=2, batch_size=8, split=0.3)
    predict(test_model, web_url)
#     save_model(cli,'thirdmodel')

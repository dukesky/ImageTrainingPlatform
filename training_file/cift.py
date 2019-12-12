#!/usr/bin/python
# Classify Image with Flow of Tensor
# Copyright 2019 Austin Yu All right reserved

# hongtaoyu@yahoo.com for license inquery
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Single Image Classfication with the model label file
# version: TF 1.10.0   TF-gpu 1.1.0   keras 2.1.5   Python 3.6



import tensorflow as tf
from tensorflow import keras
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
def loadUrl(imgUrl):
    return Image.open(imageUrl)



## Cift need to change into a new training model
class Cift:             # designed new model, based on the new input image, rather than built model
    def __init__(self, modelFileName, labelFileName, imageSize):
        self.model = tf.keras.models.load_model(modelFileName, custom_objects=None, compile=False)
        with open(labelFileName, 'r') as f:
            labelJson = json.load(f)
        self.labelArr = ['' for _ in range(len(labelJson))]
        for key in labelJson:
            self.labelArr[int(key)] = labelJson[key][1]
        self.mImageSize = imageSize
    def classify(self, imageFile):
        startTime = time.time()
        if imageFile.startswith('http'):
            img = loadImageUrl(imageFile)
        elif os.path.isfile(imageFile):
            img = loadImageFile(imageFile)
        else:
            return {"file":imageFile, "message":"Not Found"}
        img = img.resize((self.mImageSize, self.mImageSize), PIL.Image.ANTIALIAS)
        img = img.convert('RGB')
        imgFData = (np.array(img, dtype=float) - 127)/255
        imgEFData = np.expand_dims(imgFData, axis=0)
        predication = self.model.predict(imgEFData)[0]
        labelIndex = np.argmax(predication)
        return json.dumps({
                'processing_time': str(time.time() - startTime),
                'confidence': str(predication[labelIndex]),
                'classification': self.labelArr[labelIndex]
                    })
## end of change part



if __name__ == '__main__':
    ci1 = Cift('MobileNet.h5', 'gum_label_1000.json', 224)
    if len(sys.argv) > 1:
        for i in range(1,len(sys.argv)):
            pred_output=ci1.classify(sys.argv[i])
            print (ci1.classify(sys.argv[i]))
    else:
        #s = ci1.classify('cat224x224.jpg')
        pred_output = ci1.classify('https://s3.amazonaws.com/gumgum-interviews/ml-engineer/cat.jpg')

        o = json.loads(pred_output)
        print (pred_output, o)
        #print (s)
        #print (ci1.classify('https://s3.amazonaws.com/gumgum-interviews/ml-engineer/cat.jpg'))


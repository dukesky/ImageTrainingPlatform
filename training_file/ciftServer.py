# a server side API collection, include classifyImage, report, usercheck, use Flask API,
# curl      ( )     http://localhost:5100/classify-image

from flask import Flask, request
from flask_restful import Api
import cift
import subprocess
import ciftCache
import ciftmodel
import requests

app = Flask(__name__)
api = Api(app)


###############################################

@app.route('/')
def rootresponse():
    return 'Hello everyone, this is the ' \
           'online trained version of cift-server, with tensorflow version: 1.10.0'


@app.route('/train-model', methods=['POST', 'GET'])
#key name:loadmodelname,
def trainimage():
# grab data from data_files
    print('1. grab data ...')
    post_data = request.get_data()  # get_data or get_json
    post_data = post_data.decode("utf-8")
    print('data was shown as: ', post_data)

    if 'https://' in post_data[0:10] and '.json' in post_data:
        load_data = requests.get(post_data).json()
        print('get data online')
    elif '.json' in post_data:
        with open(post_data) as f:
            load_data = json.load(f)
        print('get data from local')
    else:
        load_data = post_data

    data, labels = ciftmodel.data_prepared_json(load_data)
    # here convert all input into json data or string type json data
    print('we get data of shape', data.shape)
    print('we get label', labels)
################################################


# load an existing model
    model_name = request.args.get('loadmodelname')
    if model_name == None:
        model_name= 'firstmodel'
    model = ciftmodel.load_model(model_name)
    print('load a model  ', model_name)
# build model if there is not exiting model
    try:
        model.summary()
    except UnboundLocalError:
        model = ciftmodel.built_model()
        print('finishing init a new model ...')
        try:
            a=model.get_layer
            print('now we get a new model')
        except UnboundLocalError:
            print('code does not work')

# train model
    print ('training ...')
    model, history = ciftmodel.train(model, data, labels,
                                     nb_epoch=2, split=0, batch_size=1, early_stop=True)
    ciftmodel.show_lost(history)
    print('training finished ...')

# predict image
    ciftmodel.predict(model, data[0])
# save trained new model: to where? AWS?

# model saved on the current ./models/  file

    ciftmodel.save_model(model,'currentmodel')
    print('model saved as "currentmodel"')

    return 'finish training sucessfully'


@app.route('/report-model',methods=['GET'])
def reportmodel():
# get test data
    post_data = request.get_data()  # get_data or get_json
    post_data = post_data.decode("utf-8")
    load_data = requests.get(post_data).json()

# load model and predict on test data
    model_name = 'currentmodel'
    model=ciftmodel.load_model(model_name)
    word = 'function of report model'

# calculate accuracy

    return word
################################################


@app.route('/classify-image', methods=['POST', 'GET'])
def classifyImage():
    data = request.get_json()
# get url of data
    imgUrl = data['image_url']
    if imgUrl is None:
        return 'need data file'
    # ca = ciftCache.cache(3600)
    # rtnV = ca.get(imgUrl)
    # ci = cift.Cift('MobileNet.h5', 'gum_label_1000.json', 224)
    # rtnV = ci.classify(imgUrl)
    # ca.set(imgUrl, rtnV)
    ca = ciftCache.cache(3600)
    rtnV = ca.get(imgUrl)
    if rtnV is None:
        ci = cift.Cift('./models/MobileNet.h5', 'gum_label_1000.json', 224)
        rtnV = ci.classify(imgUrl)
        #rtnV = subprocess.check_output(['./cift.py', imgUrl])
        ca.set(imgUrl, rtnV)
    return rtnV


@app.route('/report', methods=['GET'])
def report():
    ca = ciftCache.cache()
    return ca.report()


@app.route('/usercheck', methods=['GET'])
def usercheck():
    secret = request.args.get('secret')
    ca = ciftCache.cache()
    name = ca.authcheck(secret)
    if name is None:
        return "User Not Found", 404
    else:
        return "Hi " + name, 200

if __name__ == '__main__':
    #print ci.classify('https://s3.amazonaws.com/gumgum-interviews/ml-engineer/cat.jpg')
    app.debug = True
    app.run(host='0.0.0.0', port=5100)

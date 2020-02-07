#!/usr/bin/python
#%% Libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from datetime import datetime
import cv2
import tensorflow as tf
import scipy.stats as sts
import pyrem as pr
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

#%%
from flask import request, url_for, jsonify
from flask_api import FlaskAPI, status, exceptions
from flask_cors import CORS, cross_origin
import random

#%% Importing model
from tensorflow import keras
model = keras.models.load_model('my_model.h5')
print("modelo importado")


#%%
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def pred(csvName):
    input_dataset_url = csvName
    inputDataset = pd.read_csv(input_dataset_url)
    X_input = inputDataset.iloc[:,:-1].values
    
    #%% Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_Input = sc.transform(X_Input)
    
    #%%
    return model.predict(X_input)
    print("Resultado arrojado")

#%%

app = FlaskAPI(__name__)
cors = CORS(app)


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['bmp', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def index():
    return "If you're using this API through the 80 port, you're on testing, lmao -> /API/image/ para subir mediante un post -> el resultado se entregará en la cabecera de respuesta"

@app.route("/reset")
def reset():
    global elemento, x

    elemento=-1
    random.shuffle(x)
    return "reinicia2"

@app.route("/API/image/", methods=['GET', 'POST'])
@cross_origin()
def result():
    if request.method == "POST":        
        #Get image from request
        if not 'file' in request.files:
            return jsonify({'error': 'no file'}), 400
        
        # Image info
        img_file = request.files.get('file')
        img_name = img_file.filename
        mimetype = img_file.content_type
        # Return an error if not a valid mimetype
        #if not mimetype in valid_mimetypes:
        #    return jsonify({'error': 'bad-type'}),406
        # Write image to static directory
        print("Ola")
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        imgLocation="UPLOAD_FOLDER/"+img_name
        img = cv2.imread(imgLocation,0)
        

        #Preprocessing
        #If image is monochromatic
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        #Else 
        #Gray scale
        
        
        #Getting atributes
        atributes=np.zeros(8)
        
        #Kurtosis
        atributes[0]=sts.kurtosis(hist)
        #Skewness
        atributes[1]=sts.skew(hist)
        #Std
        atributes[2]=np.std(hist)
        #Range
        atributes[3]=np.ptp(hist)
        #Median 
        atributes[4]=np.median(hist)
        #Geometric_Mean 
        atributes[5]=geo_mean(hist)
        #Hjorth
        x,comp, mor= pr.univariate.hjorth(hist)
        #Mobility 
        atributes[6]=mor
        #Complexity
        atributes[7]=comp
        
        ##Saving image atributes in csv -> guardo nombre con timestamp en csv/$csvName
        ##If 1 -> save csv with timestamp
        csvName=str(datetime.now()).split(" ")[0]+"_"+str(datetime.now()).split(" ")[1]+".csv"
        saveCsv="csv/"+csvName
        numpy.savetxt(saveCsv,atributes, delimeter=",")
        result= pred(saveCsv)
        #result=random.uniform(0,1) 
        data={'result': result} 
        
        # Delete image when done with analysis
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
       
        return jsonify(data), 200 #if no hay error, else jaja
    return("ERROR: Holi, debe utilizar POST con su imagen y leer la respuesta, jajasalu2","400")

#%%
#OJO, SOLO PARA PRUEBAS ES EL PUERTO 80
#app.run(host="0.0.0.0", port=2012)
app.run(host="0.0.0.0", port=2012)
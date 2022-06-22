from importlib.resources import files
from flask import Flask, render_template, request
import pickle
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from matplotlib import image
import pytesseract as tess
from PIL import Image,ImageFont, ImageDraw
import cv2
from bs4 import BeautifulSoup
import requests
from keras.utils import load_img,img_to_array, to_categorical

#load model
file= open("python\static\model\model_pickle",'rb')
model=pickle.load(file)
model1=load_model('python\model_fix_224.h5')


# Khởi tạo Flask Server Backend
app = Flask(__name__)
# Apply Flask CORS
app.config['UPLOAD_FOLDER'] = "python\static"

# xu ly request up file anh
@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "GET":
        return render_template("web_nhan_dien.html")
    if request.method == "POST":
        image = request.files['file']
        classes={ 0:'NTV',1:'TV'}
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(path_to_save)
        img=cv2.imread(path_to_save)
        img=load_img(path_to_save,target_size=(224,224))
        img=img_to_array(img)
        img=img.reshape(1,224,224,3)
        msg1=classes[np.argmax(model1.predict(img))]
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                
        cv2.imwrite(path_to_save, img)    
        if msg1 == "TV":
            msg2= "Tiếng Việt"
        else:
            msg2="Không là Tiếng Việt"        
        return render_template("web_nhan_dien.html", user_image = image.filename, msg=msg2)
    return render_template("web_nhan_dien")

# xu ly request up link html
@app.route('/result', methods=['GET', 'POST'])
def apply_student():    
    if request.method == "GET":
        return render_template("web_nhan_dien.html")
    if request.method == "POST":
        output=request.form.to_dict()
        text= output["link_bai_bao"]
        url= requests.get(str(text)).text.encode('utf8')
        soup = BeautifulSoup(url, 'lxml')
        text_bb=soup.find('p').get_text()
        msg_tran=model.predict([text_bb])
        return render_template("web_nhan_dien.html",vanban_trich=text_bb,msgg=msg_tran[0])
    return render_template("web_nhan_dien")       


#start server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='1473',debug=True)


    

from tkinter import Frame
import cv2
import pickle
from keras.models import load_model
from keras.utils import load_img,img_to_array, to_categorical
import pytesseract as tess
file= open("python\static\model\model_pickle",'rb')
model=pickle.load(file)
tess.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'


cap=cv2.VideoCapture(0)

while True:
    ret,frame =cap.read()
    text_tran=tess.image_to_string(frame, lang='vie')
    text_predict= model.predict([text_tran])
    print(text_predict)
    cv2.putText(frame,'predict:',(10,30),cv2.FONT_HERSHEY_PLAIN,1,(50,50,255),2) 
    if text_predict =="TV":
        cv2.putText(frame,text_predict[0],(80,30),cv2.FONT_HERSHEY_PLAIN,1,(50,50,255),2) 
    else:
         cv2.putText(frame,"Not TV",(80,30),cv2.FONT_HERSHEY_PLAIN,1,(50,50,255),2)

    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

cap.realease()
cap.destroyAllWindows()
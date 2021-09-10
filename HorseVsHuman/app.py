from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
preDic={1:'Human',0:'horse'}
model=load_model('E:\WebApplication\HorseVsHuman\static\horseVShumanInceptionV3.h5')
model.make_predict_function()
def predictFun(imgPath):
    img=image.load_img(imgPath,target_size=(224,224))
    img=image.img_to_array(img)/255
    img=img.reshape(1,224,224,3)
    predArr=model.predict(img)
    if predArr[0][0]<0.5:
        print('horse: ',predArr[0][0])
        p=0
    else:
        print('Human: ',predArr[0][0])
        p=1
    return preDic[p]
    
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')
@app.route('/submit',methods=['GET','POST'])
def submit():
    if request.method=='POST':
        imag=request.files['my_image']
        imag_path="static/"+imag.filename
        imag.save(imag_path)
        p=predictFun(imag_path)
    return render_template('index.html',prediction=p,imag_path=imag_path)
if __name__ == '__main__':
    app.run(debug=True)







from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import base64
import json
import tensorflow as tf
import numpy as np
import base64
import re
import cv
import matplotlib.pyplot as plt
import face_recognition
from PIL import Image, ImageDraw
import glob
import os
import pathlib




app = Flask(__name__)
app.secret_key = 'lmao'

model = tf.keras.models.load_model("static/models/univeral_emotion_model.h5")

def parse_image(imgData):
    print(imgData)
    imgstr = re.search(b"base64,(.*)", imgData).group(1)
    img_decode = base64.decodebytes(imgstr)
    with open("output.jpg", "wb") as file:
        file.write(img_decode)
    return img_decode

@app.route("/upload/", methods=["POST"])
def upload_file():
    parse_image(request.get_data())
    image = face_recognition.load_image_file('output.jpg')
    face_locations = face_recognition.face_locations(image)

    j = 1
    predictions = ''
    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        # pil_image.show()

        # path to save file
        pil_image.save('new_image' + str(j)+'.jpg')

        im = tf.io.read_file('new_image' + str(j)+'.jpg')
        image = tf.image.decode_jpeg(im, channels=3)
        image = tf.image.resize(image, [192,192])
        image = image / 255.0
        image = 2*image - 1
        probabilites = model.predict(tf.expand_dims(image,0))
        prediction = np.argmax(probabilites, axis=1)[0]
        real_money = {0: 'Happy', 1: 'Dont happy'}
        j = j + 1
        predictions = predictions + str(real_money[prediction])

    return predictions





# data_root = pathlib.Path('static/img/happy')
# img = data_root.glob('*')
#
# save_root = pathlib.Path('static/img/final')
#
# j = 1
#
# for i in img:
#
#     image = face_recognition.load_image_file(i)
#     face_locations = face_recognition.face_locations(image)
#
#     for face_location in face_locations:
#         top, right, bottom, left = face_location
#
#         face_image = image[top:bottom, left:right]
#         pil_image = Image.fromarray(face_image)
#         # pil_image.show()
#
#         # path to save file
#         s = 'static/img/final/' + 'happy' + str(j)
#         pil_image.save(f'{s}.jpg')
#
#         j += 1




@app.route('/')
def index():
	return render_template('frontend/index.html')


@app.route("/predict", methods = ["GET"])
def predict():
    return render_template('frontend/predict.html')


@app.route('/test/', methods=["GET", "POST"])
def abc_test():
    if request.method=='POST':
        print("It is a POST request")
        try:
            print(request.form['PHOTO'])
        except Exception as e:
            print(('exception 1:', e))
        try:
            print(request.files['PHOTO'])
        except Exception as e:
            print(('exception 2:', e))
        try:
            filename=photos.save(request.files['PHOTO'])
            print("filename")
            picture=unicode(filename)
            current_user.picture=picture
            db.session.commit()
            return redirect(url_for('abc_test'))
        except Exception as e:
            print(('exception 3:', e))
        return 'DID NOT SAVE IMAGE'
    print("did not POST")
    return render_template('abc_test.html')

if __name__ == '__main__':
	app.debug=1
	app.run()

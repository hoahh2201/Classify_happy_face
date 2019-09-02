# Fansipan_W7_READ.ME

## INTRODUCTION
### 1. ABOUT THE PROJECT:
Encourage by **Facebook tag**, we are trying to build a simple application that can tell about the **emotional states** of people.

To accomplish that, we collected images from Google - with 2 main emotions: **Happy** and **Not Happy**. Then we train a model using these images, add that model to Flask app and then deploy it to google cloud for public share.
### 2. PROJECT DEPLOYMENT

![](https://scontent.fsgn2-2.fna.fbcdn.net/v/t1.0-9/69838875_2344903755826306_6637809253740445696_o.jpg?_nc_cat=103&_nc_oc=AQlGOBUaBhxca-hJEfJms_XCTTo-8KtgQzLSW--TRAWadQ4fZ55RC-Iv5q5C33mwxKw&_nc_ht=scontent.fsgn2-2.fna&oh=5d158a6466cf59f374527a2f4bbd5bd8&oe=5DC884F6)

#### SET UP ENVIRONMENT:
- [FLASK](https://flask.palletsprojects.com)
- [VSCODE](https://code.visualstudio.com/docs/setup/setup-overview)
#### IMAGES PREPROCESSING & MODEL TRAINING:
- Preprocessing using **PIL**, **os** and **face_recognition** libraries.
- Training model using **tensorflow 2.0.0**
#### APP DEPLOYMENT:
- Set up google cloud account.
- Enable billing (which give you a free tier of 300 USD)
- Deploy app.

### 3. RESULTS
- Our app is available [here](test-gcloud-290191.)

## BUIDLING EMOTION RECOGNITION APPLICATION

### Collect data & Preprocessing
- The guide for download images from google can be found [here](https://hackmd.io/8McWB9l9S-K58OxSiLM65A)
- Since we only need the face for this task, we gonna cut the image with the help of face_recognition library of Python.

```python=
import face_recognition
from PIL import Image, ImageDraw
import glob
import os
import pathlib

data_root = pathlib.Path('static/img/happy')
save_root = pathlib.Path('static/img/final')
img = data_root.glob('*')

j = 1

for i in img:

    image = face_recognition.load_image_file(i)
    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

        # path to save file
        s = 'static/img/final/'+ 'happy'+str(j)
        pil_image.save(f'{s}.jpg')
        
        j += 1
```
### Train model
This should be the easiest part since we used the build-in model from keras. This process required tensorflow 2.0.0 so remember to upgrade your tensorflow by:
```python
pip3 install tensorflow==2.0.0-beta1
```
Then we define a function for loading the data's paths and labels.
```python=
def load_image_dataset(file_path):

    # get all class
    all_class = {}
    for index, folder in enumerate(sorted(listdir(file_path))):
        if folder != ".ipynb_checkpoints":
            all_class[folder] = index - 1
        
    all_class_path = [join(file_path, path) for path in sorted(listdir(file_path))]
    
    im_path = [[join(cl, path) for path in listdir(cl) if path.rsplit('.')[-1].lower() in ['jpg', 'jpeg']] for cl in all_class_path]
    im_label = [[all_class[cl.rsplit('/')[-1]] for path in listdir(cl) if path.rsplit('.')[-1].lower() in ['jpg', 'jpeg']] for cl in all_class_path]
    
    # Get all image paths
    im_path_final = []
    for paths in im_path:
        for path in paths:
            im_path_final.append(path)

    # Get all image labels
    im_label_final = []
    for labels in im_label:
        for label in labels:
            im_label_final.append(label)
            
    return im_path_final, im_label_final, all_class
```

Load the image paths, labels and save it to **paths** and **labels**:
```python
# call load_image_dataset function and pass in the path of the root folder where your images stored.
paths, labels, all_class = load_image_dataset(file_path)
```

Then we define some function to read & preprocessing image from image's path.

```python=
# Function that take care of preprocessing image

def load_and_preprocess_image(path):
    file = tf.io.read_file(path)
    image = tf.image.decode_jpeg(file, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    image = 2*image-1  # normalize to [-1,1] range
    return image

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label
```
Finally, we put all the images and labels into **tf Dataset** to help the training process much easier.

```python=
# Put the image's paths and labels of train and test set to tensorflow dataset.
ds = tf.data.Dataset.from_tensor_slices((paths, labels))

#using map function on ds with load_and_preprocess_from_path_label function - which return the image itself and its label.
image_label_ds = ds.map(load_and_preprocess_from_path_label)
```

Now we got all the images and labels stored in **image_label_ds**, it's time to build a model to train our data.

```python=
# Load the pre-train model MobileNetV2 from keras
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

# Create CNN model
cnn_model = keras.models.Sequential([
    mobile_net,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])
```
Apply batch on training data, which help reduce training time significantly. Give it a new name **train_ds**

```python=
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = image_label_ds.shuffle(buffer_size = len(labels))
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
```
Well, we got all the thing we need. Now let's train the model.
```python=
# Compile CNN-model
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
```
```python=
# Training the model with the train_ds above.
steps_per_epoch=tf.math.ceil(len(labels)/BATCH_SIZE).numpy()
cnn_model.fit(train_ds, epochs=1, steps_per_epoch=steps_per_epoch)
```
**RESULT:**
- The model get 91 accuracy scores on training set.
- This model would be save as .h5 file for later use in app deployment part.

### Build an app
- Follow this [video](https://www.youtube.com/watch?time_continue=191&v=QjtW-wnXlUY) **from 0:00 to 4:02** for setting up a new flask project.
- After that, type "**code .**" to open visual code studio (or vscode for short).
- Continue follow the video exactly **from 4:03**, but this time we do it in vscode environment - not in sublime (which the video does).  
- After finish debuging the simple task, we gonna put in some file that help us to handle the front-end and back-end task. I will not cover this task in this project, but you can download it from [here]() and put them in your flask app (for fun, or testing!!!).
## CONCLUSION:




# Classify_happy_face

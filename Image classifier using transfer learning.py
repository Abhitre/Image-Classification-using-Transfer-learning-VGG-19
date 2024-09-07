from flask import Flask, request, render_template
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Load the VGG19 model and weights from ImageNet
model = VGG19(weights='imagenet')

# Function to preprocess and predict the image
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

# Flask routes
@app.route('/', methods=['GET'])
def index():
    return '''
        <html>
        <body>
            <h1>Upload Image for Prediction</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
        </html>
    '''

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0][0][1])
        return result

if __name__ == '__main__':
    app.run(debug=True)


  




 

       

    
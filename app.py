from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

modelpath = "SC-MobileNetV2.h5"
MODEL = tf.keras.models.load_model(modelpath)

classnames = {
    4: ('mel', 'melanoma'),
    6: ('vasc', 'vascular lesion'),
    2: ('bkl', 'benign keratosis-like lesions'),
    1: ('bcc', 'basal cell carcinoma'),
    5: ('nv', 'melanocytic nevi'),
    0: ('akiec', 'Actinic keratoses'),
    3: ('df', 'dermatofibroma')
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def save_processed_images(image):
    kernel = np.ones((5, 5), np.uint8)
    data = image
    grayScale = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(data, thresh2, 1, cv2.INPAINT_TELEA)
    # temp = cv2.resize(dst, (256, 256))
    # temp_rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    # cv2.imwrite('abdo.jpg', temp_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return dst

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            image = file.read()
            image = np.array(Image.open(BytesIO(image)).resize((64, 64)))
            image = save_processed_images(image)
            image = np.expand_dims(image, 0)
            predictions = MODEL.predict(image)
            
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_class_labels = [classnames[i] for i in top3_indices]
            top3_probabilities = [predictions[0][i] for i in top3_indices]

            response = {
                "predictions": [
                    {"class": top3_class_labels[i][1], "probability": float(top3_probabilities[i])}
                    for i in range(3)
                ]
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()

from flask import Flask, render_template, request, jsonify
from datetime import datetime
from cv2 import imread
import json, base64, cv2, os
import numpy as np
from matplotlib import pyplot as plt

from NeuralNetwork.image_network import import_network_from_file, predict_number, train_custom_dataset, export_network_to_file
from NeuralNetwork.data_generator import CustomGenerator


app = Flask(__name__,
            template_folder='./src',
            static_folder='public',
            static_url_path='')


# Global Variables
current_date = datetime.today().strftime('%b-%d-%Y')

# Helper functions
def save_img(img_base64, folder_name, userID=None, data_number=None, training_data=False):
    # setting date variables
    current_time = round(datetime.now().timestamp())

    # setting up folder directories based on preference
    current_dir = os.getcwd()
    current_folder_dir = f'{current_dir}/NeuralNetwork/{folder_name}'
    if not os.path.isdir(current_folder_dir):
        os.mkdir(current_folder_dir)

    current_date_dir = f'{current_dir}/NeuralNetwork/{folder_name}/{current_date}'
    if not os.path.isdir(current_date_dir):
        os.mkdir(current_date_dir)

    if data_number and training_data and userID:
        current_user_dir = f'{current_dir}/NeuralNetwork/{folder_name}/{current_date}/{userID}/'
        if not os.path.isdir(current_user_dir):
            os.mkdir(current_user_dir)
        
        folder_dir = f'{current_user_dir}/{data_number}'
    else:
        folder_dir = f'{current_dir}/NeuralNetwork/{folder_name}/{current_date}'
    
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
    
    file_name = f'{current_time}.jpg' # include img file type here
    file_path = os.path.join(folder_dir, file_name)
    
    # converting from base64 to img file if your choice
    nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    # saving img on server
    cv2.imwrite(file_path, img)

    # return path of saved img file
    return file_path


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # setting img variable
        imgData = request.json['data']
        imgData = imgData.split(',')[1]
        userID = request.json['userid']

        # saving img to server
        file_path = save_img(imgData, 'user_numbers')

        # import neural network and predicting user's data
        network_filename = userID
        network = import_network_from_file(network_filename) # loading neural network from server
        prediction = predict_number(network, file_path)

        return jsonify(prediction)

@app.route('/saveusertraindata', methods=["POST"])
def user_training_data():
    if request.method == "POST":
        # setting img variable
        userID = request.json['userid']
        imgData = request.json['data']
        imgAnswer = request.json['dataResult']
        imgData = imgData.split(',')[1]
        
        if imgData and imgAnswer:
            # saving img to server
            file_path = save_img(imgData, 'user_training_data', userID, data_number=imgAnswer, training_data=True)

            return jsonify('User training data saved successfully')
        else:
            return jsonify('Unable to save training data')

@app.route('/trainuserdata', methods=["POST"])
def train_user_data():
    if request.method == "POST":
        # obtain latest user data
        data_name = request.json['userid']
        data_dir = 'user_training_data'
        custom_data = CustomGenerator(data_dir, data_name)

        # save training data
        custom_data.save_data_to_file()

        # import neural network
        network = import_network_from_file(data_name)

        # obtaining saved training data
        training_imgs, training_results = custom_data.load_data_from_file(training_only=True) # using current date as data name

        # train neural network
        epochs = 3
        batch_size = len(training_imgs) if len(training_imgs) < 30 else max(round(len(training_imgs) * 1/3), 1)
        train_custom_dataset(network, training_imgs, training_results, epochs, batch_size=batch_size)

        # save trained network as a new file
        export_network_to_file(network, data_name)

        return jsonify("Model training completed! You may now test my new improved AI.")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8003, debug=True)
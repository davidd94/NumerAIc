import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
from datetime import datetime


""" *** Neural Network Settings """
EPOCHS = 2
BATCH_SIZE = 100
# Data size of 500-1000 should typically have a learning rate of 0.001
# My test data sample is only 100 so I've used learning rate of 0.0001
LEARNING_RATE = 0.001

IMG_SIZE = 28
MODEL_PATH = f'/Users/davidd/Web Apps/NumerAIc/NeuralNetwork'

# Loss and optimizer
#loss = nn.MSELoss()
loss = nn.CrossEntropyLoss()


class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential( # this allows us to create sequentially ordered layers in our network
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), # input of a single grayscale img, output (channels) of 32,
                                                                  # kernel is 5x5 window or filter as it goes through each channel's img features,
                                                                  # stride shifts the kernel or window one to the right
                                                                  # padding is calculated as 2 with the given input (see docu for eq)
            nn.ReLU(), # activation function via Rectified Linear Unit
            nn.MaxPool2d(kernel_size=2, stride=2)) # padding defaults to 0
            # the output from layer1 will be 32 channels of 14x14 'images' due to halfing the size from stride = 2

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # input of 32 channels from layer1, outputting 64 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            # the output from layer2 will be 64 channels of 7x7 'images' due to halfing the size from stride = 2

        self.drop_out = nn.Dropout() # to avoid over-fitting within the model
        self.fc1 = nn.Linear(7*7*64, 1000) # input of 7x7 images size multiplied by 64 channels, output 1000
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1) # we MUST flatten the data prior inserting into Linear network.
                                    # Flattening data dimensions from 7 x 7 x 64 channels into 3136 x 1
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        # NOTE: Softmax is NOT included for Mean Squared Error (MSE) but IS included for Cross Entropy
        #meep = nn.Softmax(dim=1)
        #return meep(x)
        return x

# Recommended to train and test your model first prior saving to file
def export_network_to_file(net, net_filename):
    if isinstance(net, ConvNeuralNetwork) is False:
        return "Unable to export network. Incorrect network type"
    
    current_date = datetime.today().strftime('%b-%d-%Y')
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    full_model_path = f'{MODEL_PATH}/{current_date}-{net_filename}.pt'
    torch.save(net.state_dict(), full_model_path)
    
    return "Successfully saved model!"

def import_network_from_file(net_filename):
    net = ConvNeuralNetwork()
    current_date = datetime.today().strftime('%b-%d-%Y')
    full_model_path = f'{MODEL_PATH}/{current_date}-{net_filename}.pt'
    if os.path.exists(full_model_path):
        net.load_state_dict(torch.load(full_model_path))
        net.eval()
        return net
    else:
        net.load_state_dict(torch.load(f'{MODEL_PATH}/network_model.pt'))
        net.eval()
        return net

def train_MNIST_dataset(net, data):
    if isinstance(net, ConvNeuralNetwork) is False:
        return "Unable to train network with MNIST dataset. Incorrect network type"
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    data_length = len(data)
    loss_list = []
    accuracy_list = []
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(data):
            # Execute forward pass
            outputs = net(images)
            calc_loss = loss(outputs, labels)
            loss_list.append(calc_loss.item()) # calc_loss is still a tensor so to extract the data we must use .item()
            
            # Back Propragation and Adam Optimizations
            optimizer.zero_grad() # zeroing the gradient prior back propragation
            calc_loss.backward() # back propragation
            optimizer.step()
            
            # Tracking Accuracy
            total = labels.size(0) # EACH EPOCH'S BATCH SIZE WHICH SHOULD BE 100
            
            _, predicted = torch.max(outputs, 1) # torch.max(tensor, axis) returns '_' as max values within each img and 'predicted' as
                                                 # max values' index number which correlates to the predicted img number.
                                                 # Each in index will represent the number 0-9, totaling 10.
                                                 # Remember that the output for each image results will output tensor length of 10 data points.
            
            correct = (predicted == labels).sum().item() # Checking prediction tensor with label tensor to see if predictions are correct.
                                                         # Sum all the correct predictions with .sum() and convert from tensor to int with .item()
            
            accuracy_list.append(correct / total)
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {calc_loss.item()}, Accuracy: {(correct/total) * 100} %')

def test_MNIST_dataset(net, data):
    if isinstance(net, ConvNeuralNetwork) is False:
        return "Unable to test network with MNIST dataset. Incorrect network type"
    correct = 0
    total = 0
    # eval() disables any drop-out o batch normalization layers within our network
    net.eval()
    
    # We do NOT want to train/optimize our data and only test our 
    # neural network's accuracy, so gradient must not be used!
    with torch.no_grad():
        for images, labels in data:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Test Accuracy from 10,000 test images: {(correct / total) * 100}%')

# MUST use DataGenerator to export the proper formatted file for these funcs
def train_custom_dataset(net, imgData, imgOutcomes, epochs, batch_size):
    if isinstance(net, ConvNeuralNetwork) is False:
        return "Unable to train network with custom dataset. Incorrect network type"
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    EPOCHS = epochs if epochs else EPOCHS
    BATCH_SIZE = batch_size if batch_size else len(imgData)
    for epoch in range(EPOCHS):
        for i in range(0, len(imgData), BATCH_SIZE): # splitting up all the data into their respective batch size using 'range'
            batch_X = imgData[i:i + BATCH_SIZE].view(-1, 1, 28, 28)
            # For Cross Entropy Loss, you must have all the batch's number results in a tensor
            batch_Y = torch.argmax(imgOutcomes[i:i + BATCH_SIZE], 1)
            # Whereas for Mean Standard Error (MSE) Loss, you must have the batch's number results in a tensor AS a ONE HOT VECTOR
            # batch_Y = imgOutcomes[i:i + BATCH_SIZE] # This format is already in a one HOT VECTOR format
            net.zero_grad()
            outputs = net(batch_X)
            calc_loss = loss(outputs, batch_Y)
            calc_loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}. Loss: {calc_loss}')

def test_custom_dataset(net, imgData, imgOutcomes):
    if isinstance(net, ConvNeuralNetwork) is False:
        return "Unable to test network with custom dataset. Incorrect network type"
    correct = 0
    total = 0
    # eval() disables any drop-out or batch normalization layers within our network
    net.eval()
    
    # We do NOT want to train/optimize our data and only test our 
    # neural network's accuracy, so gradient must not be used!
    with torch.no_grad():
        for i in range(len(imgData)):
            batch_X = imgData[i].view(-1, 1, 28, 28)
            # For Cross Entropy Loss, you must have all the batch's number results in a tensor
            batch_Y = torch.argmax(imgOutcomes[i])
            # Whereas for Mean Standard Error (MSE) Loss, you must have the batch's number results in a tensor AS a ONE HOT VECTOR
            # batch_Y = imgOutcomes[i:i + BATCH_SIZE] # This format is already in a one HOT VECTOR format
            number_answer = torch.argmax(imgOutcomes[i])
            outputs = net(batch_X)
            predicted_num = torch.argmax(outputs).item()
            
            if predicted_num == number_answer:
                correct += 1
            total += 1
            
        print(f'Total: {total}')
        print(f'Test Accuracy from outside images samples: {(correct / total) * 100}%')

def predict_number(net, imgFilePath):
    # load img file and convert to gray scale. Color doesn't help determine number and gray increases speed
    img = cv2.imread(imgFilePath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize img to 28x28 if not by default
    nparray = np.array(img) # convert img to numpy array
    imgData = torch.Tensor(nparray).view(-1, 1, 28, 28) # convert to tensor format as required by neural network configuration
    
    with torch.no_grad():
        output = net(imgData.view(-1, 1, 28, 28))
        predicted_number = torch.argmax(output).item()
        print(predicted_number)
        return predicted_number
    return 'An error as occurred during prediction...'

def retrain_with_custom_data(net):
    pass
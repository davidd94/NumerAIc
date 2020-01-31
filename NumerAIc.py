import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms, datasets
from matplotlib import pyplot as plt


train = datasets.MNIST("", train=True,
                           download=False,
                           transform=transforms.Compose([transforms.ToTensor()])) # converts data into Tensor

test = datasets.MNIST("", train=False,
                          download=False,
                          transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True) # batch size (8-64) is usually in base-8
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# FOR EDUCATIONAL NOTES ONLY
#random_data = torch.rand([28, 28]) # randomly generates a tensor with size 28x28
#random_data = random_data.view(-1, 28*28) # -1 means we can pass in any amt of arrays where 1 is only 1 batch of 28x28 can be inserted


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._to_linear = None

        self.conv1 = nn.Conv2d(1, 32, 5) # (1 img input, 32 node output, 5 kernal size of 5x5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(28, 28).view(-1, 1, 28, 28)
        self.convs(x)

        """define fully connected (fc) Layers using nn.Linear(inputSize of previous layer or data size if 1st,
        output layer aka 'hidden layer' size of you choice)
        """
        self.fc1 = nn.Linear(self._to_linear, 512) # inital data img size of 28x28 and desired output of 64
        #self.fc2 = nn.Linear(64, 54) # previous layer size input of 64 and desired output of 54 (can be any size..)
        #self.fc3 = nn.Linear(54, 27)
        self.fc4 = nn.Linear(512, 10) # outcome results is 10 for numbers 0-9
    
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # max pooling over 2x2 windows totaling 4 windows
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        
        return x

    def forward(self, x): # inherited class runs this method when instanitated
        """
        Forwarding or passing through each layer.. using an activation function relu (Rectified Linear Unit).
        Activation function detects if there the neuron node is 'fired' like real neurons.
        """
        x = self.convs(x)
        x = x.view(-1, self._to_linear)

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x) # do not need activation function here as the end results will be what we will examine

        return F.log_softmax(x, dim=1)


net = NeuralNetwork()

"""
net.parameters() is anything that is adjustable in our model
lr is the learning rate which dictates the size of each left 
and right 'bounce' that gets smaller over each learning iteration.
    Loss % lower for each learning iteration completes. Each iteration
will adjust the weights of each links within the 'hidden layers' to
achieve a lower loss %.
"""
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 1 # the req amount of FULL training and testing session


def train_on_cpu(net):
    for epoch in range(EPOCHS):
        for data in train_set:
            X, Y = data
            print(X)
            print(Y)
            net.zero_grad() # zero the gradient before doing back propragation
            output = net(X.view(-1, 1, 28, 28)) # formats the tensor
            #loss = nn.MSELoss() # vector values must use mean square error
            loss = F.nll_loss(output, Y) # scalar value so must use this
            loss.backward() # back propragation
            optimizer.step() # adjust the weights for us

        print(loss)

def test_on_cpu(net):
    correct = 0
    total = 0

    # We do NOT want to train/optimize our data and only test our 
    # neural network's accuracy, so gradient must not be used!
    # Normally we should be using out of sample data to test
    with torch.no_grad():
        for data in test_set:
            X, Y = data
            output = net(X.view(-1, 1, 28, 28))

            for idx, i in enumerate(output):
                if torch.argmax(i) == Y[idx]:
                    correct += 1
                total += 1
    
    print("Accuracy: ", round(correct/total, 3))


if __name__ == "__main__":
    train_on_cpu(net)
    test_on_cpu(net)
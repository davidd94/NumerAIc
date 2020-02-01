import os, cv2, torch, torchvision
import numpy as np
from torchvision import transforms, datasets


class MNISTGenerator():
    DATA_DIR = './'
    BATCH_SIZE = 100
    train_loader = None
    test_loader = None

    def export_MNIST_data(self):
        # data transformation into specific Pytorch Tensor data type used in Pytorch for all various data and weight operations
        # Compose() normalizes the data into ranges of -1 to 1 or 0 to 1. Means of 0.1307 and Standard deviation of 0.3081
        # Note: Since we're using MNIST dataset that is grayscale (single channel) we do NOT necessarily need to provide MEANS and STD
        # Note: For color images, we need MEANS and STD to be applied to each of the 3 channels (each channel for RGB spectrum)
        transform_data = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root=self.DATA_DIR, train=True, transform=transform_data, download=True)
        test_dataset = torchvision.datasets.MNIST(root=self.DATA_DIR, train=False, transform=transform_data, download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        self.train_loader = train_loader
        self.test_loader = test_loader

        return train_loader, test_loader

class CustomGenerator():
    IMG_SIZE = 28
    DATA_DIR = os.getcwd() + '/custom_numbers/'
    LABELS = [0,1,2,3,4,5,6,7,8,9]
    dataset = []
    
    def create_data(self):
        for label in self.LABELS:
            data_path = os.path.join(self.DATA_DIR, str(label))
            for f in os.listdir(data_path):
                try:
                    path = os.path.join(data_path, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) # image is already 28x28 but for usage on any others in future
                    self.dataset.append([np.array(img), np.eye(10)[self.LABELS[label]]]) # setting the numpy 'eye' outcomes for each img
                except Exception as e:
                    pass
        
        np.random.shuffle(self.dataset)
        np.save(self.DATA_DIR + "NumberImgData.npy", self.dataset)
    
    def export_data(self, train_percent):
        outer_sample = np.load("./custom_numbers/NumberImgData.npy", allow_pickle=True)
        X = torch.Tensor([i[0] for i in outer_sample]).view(-1, 28, 28) # separating the img values into the proper format
        Y = torch.Tensor([(i[1]) for i in outer_sample]) # separating the img answers
        
        test_percent = round(1 - (train_percent if train_percent < 1 and train_percent > 0 else 0.9))
        test_val = int(len(X) * test_percent)

        train_imgs = X[:-test_val]
        train_outcomes = Y[:-test_val]

        test_imgs = X[-test_val:]
        test_outcomes = Y[-test_val:]

        return train_imgs, train_outcomes, test_imgs, test_outcomes
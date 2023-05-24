import glob
from zipfile import ZipFile

#import pandas
#import opendatasets
import torch
from kaggle import KaggleApi
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.models as models
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization

import cv2

import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class CustomTrainSet():
    def __init__(self, imgs_path, class_name):
        #for train set, start at 0 and go to 60
        count = 0
        self.imgs_path = imgs_path
        file_list = glob.glob(self.imgs_path + "*")
        self.trainData = []
        for class_path in file_list:
            class_name = class_name
            for img_path in glob.glob(class_path + "/*.jpeg"):
                if count<=60:
                    self.trainData.append([img_path, class_name])
                    count = count + 1
                else:
                    break
        self.class_map = {"basking": 0, "blue": 1, "bull": 2, "lemon": 3, "mako": 4,  "thresher": 5, "tiger": 6, "whale": 7,  "whitetip": 8}
        self.img_dim = (28, 28)

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        img_path, class_name = self.trainData[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class CustomTestSet():
    def __init__(self, imgs_path, class_name):
        #for test set, start at 61 and go to 71
        count = 0
        self.imgs_path = imgs_path
        file_list = glob.glob(self.imgs_path + "*")
        self.testData = []
        for class_path in file_list:
            class_name = class_name
            for img_path in glob.glob(class_path + "/*.jpeg"):
                if count<=60:
                    count = count+1
                elif count>60 or count<71:
                    self.testData.append([img_path, class_name])
                    count = count+1
                else:
                    break
        #self.class_map = dictionary, convert string of class name to number
        #image dimension is the size that we will resize all the images to, so that they all have the same size.
        self.class_map = {"basking": 0, "blue": 1, "bull": 2, "lemon": 3,"mako": 4, "thresher": 5, "tiger": 6, "whale": 7, "whitetip": 8}
        self.img_dim = (28, 28)

    def __len__(self):
        return len(self.testData)

    def __getitem__(self, idx):
        img_path, class_name = self.testData[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

def getDataLoaders(filename):
    # Divide datas into train and test
    # Create data loaders for the train sets.
    basking_trainSet = CustomTrainSet(filename+"/basking_resized", "basking")
    blue_trainSet = CustomTrainSet(filename+"/blue_resized", "blue")
    bull_trainSet = CustomTrainSet(filename+"/bull_resized", "bull")
    lemon_trainSet = CustomTrainSet(filename+"/lemon_resized", "lemon")
    mako_trainSet = CustomTrainSet(filename+"/mako_resized", "mako")
    thresher_trainSet = CustomTrainSet(filename+"/thresher_resized", "thresher")
    tiger_trainSet = CustomTrainSet(filename+"/tiger_resized", "tiger")
    whale_trainSet = CustomTrainSet(filename+"/whale_resized", "whale")
    whitetip_trainSet = CustomTrainSet(filename+"/whitetip_resized", "whitetip")

    train_sets = torch.utils.data.ConcatDataset([basking_trainSet, blue_trainSet, bull_trainSet,
                                                 lemon_trainSet, mako_trainSet, thresher_trainSet,
                                                 tiger_trainSet, whale_trainSet, whitetip_trainSet])

    #create the dataloaders for the test sets
    basking_testSet = CustomTestSet(filename+"/basking_resized", "basking")
    blue_testSet = CustomTestSet(filename+"/blue_resized", "blue")
    bull_testSet = CustomTestSet(filename+"/bull_resized", "bull")
    lemon_testSet = CustomTestSet(filename+"/lemon_resized", "lemon")
    mako_testSet = CustomTestSet(filename+"/mako_resized", "mako")
    thresher_testSet = CustomTestSet(filename+"/thresher_resized", "thresher")
    tiger_testSet = CustomTestSet(filename+"/tiger_resized", "tiger")
    whale_testSet = CustomTestSet(filename+"/whale_resized", "whale")
    whitetip_testSet = CustomTestSet(filename+"/whitetip_resized", "whitetip")

    test_sets = torch.utils.data.ConcatDataset(
        [basking_testSet, blue_testSet, bull_testSet, lemon_testSet,
         mako_testSet, thresher_testSet, tiger_testSet, whale_testSet,
         whitetip_testSet])

    return train_sets, test_sets
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Define model
class OBD(nn.Module):
    def __init__(self):
            super(OBD, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=64,
                    kernel_size=5,
                    stride=1,
                    padding=1
                ),
                #input layer
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            )

            #average pooling layer comes second - subsampling
            #second layer
            #self.pool_1 = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2), padding = 0),

            BatchNormalization()
            #3rd layer
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                in_channels=64,
                out_channels=12,
                kernel_size=5,
                stride=1,
                padding=1
                ),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            )
            # local averaging
            #2-to-1 subsampling
            #fourth layer
            nn.Dropout(0.2)
            #self.pool_2 = nn.AvgPool2d(kernel_size=(2,2), stride = (2,2), padding = 0)
            # fully connected layer, output 10 classes
            nn.Flatten()
            #need to flatten
            self.out = nn.Linear(300, 9)
            #using linear = weights affected, everything connected

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv1(x)
        #x = self.pool_1(x)
        x = self.conv2(x)
        #x = self.pool_2(x)
        # flatten the output of conv2 to (batch_size, 12 * 4 * 4)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        # return output, x    # return x for visualization
        return output

def main():
    print("CLASSIFIER 2:")
    filename = "sharks/sharks"
    batch_size = 64

    #downloadBool = input("Do you want to download the datasets?")
    downloadBool = "N"
    if downloadBool == "Y":
        api = KaggleApi()
        api.authenticate()
        print("Downloading files...")
        api.dataset_download_files('clairefielden/sharks')
        print("Unzipping files...")
        zf = ZipFile('sharks.zip')
        zf.extractall('sharks')  # save files in selected folder
        filename = "sharks/sharks"
        zf.close()

    trainingSet, validationSet = getDataLoaders(filename)
    train_dataloader = DataLoader(trainingSet, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(validationSet, batch_size=batch_size, shuffle=True)

    model = OBD().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    print("Done!")

    torch.save(model.state_dict(), "model2.pth")
    print("Saved PyTorch Model State to model2.pth")
    #save the model

    model = OBD()
    model.load_state_dict(torch.load("model2.pth"))
    model.eval()
    #retrieve the model

    print(model)

    classes = [
        "basking",
        "blue",
        "bull",
        "lemon",
        "mako",
        "thresher",
        "tiger",
        "whale",
        "whitetip"
    ]

    #obtain input image
    #img = input("Please enter a filepath: ")
    test_img = Image.open(filename+"/lemon_resized/lemon (7).jpeg")
    #THROW EXCEPTION IF NOT THERE
    test_img_data = np.asarray(test_img)
    plt.imshow(test_img_data)
    plt.show()

    # model expects 28x28 greyscale image
    transform = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor()
    ])

    # standard ImageNet normalization
    transform_normalize = transforms.Normalize((0.5,),(0.5,))

    transformed_img = transform(test_img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

    with torch.no_grad():
        pred = model(input_img)
        predicted = classes[pred[0].argmax(0)]
        print(f'Classifier: {predicted}')


if __name__ == "__main__":
    main()
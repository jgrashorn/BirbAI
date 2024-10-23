import torch
import torchvision.transforms as transforms
import torchvision.models as tvModels
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import PIL
from matplotlib import pyplot as plt
import numpy as np

import cv2

import pandas as pd
import os

from sklearn.metrics import confusion_matrix
import seaborn as sn

def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    valLoss = float(correct / total)
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
    return valLoss

class BirdDataset(torch.utils.data.Dataset): # inheritin from Dataset class
    def __init__(self, pdDataset, root_dir="", transform=None):
        self.annotation_df = pdDataset
        self.root_dir = root_dir # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df) # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        # print(os.path.join(self.root_dir, self.annotation_df.iloc[idx, 1]))

        image_path = os.path.join(self.root_dir, self.annotation_df.iloc[idx, 1]) #use image path column (index = 1) in csv file
        image = cv2.imread(image_path) # read image by cv2
        #print("Imshape")
        #print(f"Before transform: {image.shape}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB for matplotlib
        class_name = self.annotation_df.iloc[idx, 2] # use class name column (index = 2) in csv file
        class_index = self.annotation_df.iloc[idx, 3] # use class index column (index = 3) in csv file
        if self.transform:
            image = self.transform(image)
        #print(f"After transform: {image.shape}")
        return image, class_index
    
if __name__ == "__main__":

    IMGSIZE = 512
    batch_size = 16
    mName = "model_temp_RN101"

    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((IMGSIZE, IMGSIZE), interpolation=PIL.Image.BILINEAR),
        ])
    
    transformResNet = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    fullData = pd.read_csv('data.csv')

    trainDS = fullData.sample(frac=0.8,random_state=159)
    testDS = fullData.drop(trainDS.index)

    classes = fullData["class_name"].unique()

    testDatasetTransformed = BirdDataset(testDS, root_dir="", transform=transformResNet)

    testloader = torch.utils.data.DataLoader(testDatasetTransformed, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    
    model = tvModels.resnet101()

    model.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=model.fc.in_features,
                out_features=240
            ),
            torch.nn.Sigmoid()
        )
    
    mState = torch.load(f"{mName}.tar",weights_only=False)
    
    model.load_state_dict(mState['model_state_dict'])
    print("Loaded model!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred = []
    y_true = []

    # iterate over test data
    i = 0
    model.eval()
    model.to(device)

    for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Feed Network

            # output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            _, output = torch.max(outputs.data, 1)
            output = output.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth
            # print(f"Label: {np.argmax(np.bincount(labels))}, avg. predicted: {np.argmax(np.bincount(output))}")
            # i+=1
            # if i%200==0:
            #     print(f"Evaluating {i}th batch")

    # constant for classes
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    
    df_cm.to_csv("ConfusionMatrix.csv")

    plt.figure(figsize = (100,100))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'LossPlots/{mName}_ConfusionMatrix.png')
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
    
# Function to train the neural network
def train_model(model, train_loader_generator, test_loader, criterion, optimizer,
                scheduler, num_epochs, device, validateAtStep=10, modelName = "model_temp", saveAtStep=10):
    model.train()
    # model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_loader = train_loader_generator()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()

        if os.path.exists(f"{modelName}.log"):
            with open(f"{modelName}.log", "r") as file:
                for row in file:
                    pass
            
            row = row.split(",")
            ep = int(row[0])
            prevLoss = float(row[1])
        else:
            with open(f"{modelName}.log", "w") as file:
                file.write("epoch,loss,validation loss\n")
            ep = 0
            prevLoss = -1
        
        trainLoss = running_loss/len(train_loader)#

        if (epoch+1) % saveAtStep == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {trainLoss:.4f}, prev: {prevLoss}, lr: {scheduler.get_last_lr()}")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss
                },f"{modelName}.tar")

        if (epoch+1) % validateAtStep == 0:
            valLoss = evaluate_model(model, test_loader, device)
        else:
            valLoss = -1

        with open(f"{modelName}.log","a") as file:
            file.writelines(f"{ep+1}, {trainLoss:.4f}")
            if valLoss > 0:
                file.writelines(f", {valLoss:.4f}")
            file.writelines("\n")

# Function to evaluate the neural network
def evaluate_model(model, test_loader, device):
    model.eval()
    # model.to(device)
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
    model.train()
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
    
def SetupReducedDataset(dataset, nData) -> BirdDataset:
    trainDSReduced = pd.DataFrame(columns=dataset.columns)
    for i in indices:
        curr_ = dataset[dataset["class_index"] == i]
        if len(curr_.index) > nData:
            rndPrm = np.random.permutation(len(curr_.index))[0:nData]
            trainDSReduced = pd.concat( [trainDSReduced, pd.DataFrame(curr_.loc[curr_.index[rndPrm],:],columns=fullData.columns)] , ignore_index=True, sort=False)
        else:
            trainDSReduced = pd.concat( [trainDSReduced, curr_] , ignore_index=True, sort=False)
    return trainDSReduced
    
if __name__ == "__main__":

    IMGSIZE = 512
    NDataPerClass = 150
    validateAtStep = 50
    saveAtStep = 5

    mName = "model_temp_RN101"

    transformResNet = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(256),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(degrees=45),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                        transforms.CenterCrop(224),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    fullData = pd.read_csv('data.csv')

    trainDS = fullData.sample(frac=0.8,random_state=159)
    testDS = fullData.drop(trainDS.index)

    numData = dict()
    indices = fullData["class_index"].unique()
    numClasses = len(indices)
    print(f"Number of classes: {numClasses}")

    for k in indices:
        nData = sum(fullData["class_index"]==k)
        nTrain = sum(trainDS["class_index"]==k)
        nTest = sum(testDS["class_index"]==k)
        numData[k] = [nData,nTrain,nTest]

    trainDatasetTransformed = lambda: BirdDataset(SetupReducedDataset(trainDS,NDataPerClass), root_dir="", transform=transformResNet)
    testDatasetTransformed = BirdDataset(testDS, root_dir="", transform=transformResNet)

    batch_size = 128
    input_size = IMGSIZE  # Example input image size

    # rndSampler = torch.utils.data.RandomSampler(trainDatasetTransformed)
    trainloader = lambda: torch.utils.data.DataLoader(trainDatasetTransformed(), batch_size=batch_size,
                                                shuffle=True, num_workers=8)

    testloader = torch.utils.data.DataLoader(testDatasetTransformed, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = tvModels.resnet101()

    model.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=model.fc.in_features,
                out_features=numClasses
            ),
            torch.nn.Sigmoid()
        )
    
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    # optimizer.to(device)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # scheduler.to(device)

    # model = torch.load(f"{mName}.mod")
    loaded = 0
    try:
        state = torch.load(f"{mName}.tar",map_location=device,weights_only=False)
        loaded = 1
    except Exception as e:
        print("Couldn't load! Continuing....")

    if loaded == 1:
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])

    #     model = SimpleCNN(input_size=input_size, num_classes=numClasses, layers=layers)

    # torch.save(model,f"{mName}.mod")

    criterion = nn.CrossEntropyLoss()

    # for g in optimizer.param_groups:
    #     g['lr'] = 0.0001

    # device = "cpu"
    # print(f"Training on device {device}")

    train_model(model, trainloader, testloader,
                criterion, optimizer, scheduler, num_epochs=1000,
                device = device, validateAtStep=validateAtStep, modelName=mName,
                saveAtStep=saveAtStep)

    # torch.save(model,"model.mod")

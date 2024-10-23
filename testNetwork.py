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

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w + 2, y + text_h + 2), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

if __name__ == "__main__":

    mName = "model_temp_RN101.tar"
    mPath = f"D:\Seafile\BirbAI\Models\{mName}"

    print(mPath)

    folderName = "testImages"

    german = 1

    sciNames = []
    gerNames = []
    engNames = []

    start = 0
    with open("aliasesFull.txt","r",encoding='utf-8') as file:
        lines = file.readlines()
        for row in lines:
            if start==0:
                start=1
            else:
                sN, gN, eN = row.split(",")
                sciNames.append(sN.strip())
                gerNames.append(gN.strip())
                engNames.append(eN.strip())

    if german:
        dispNames = gerNames
    else:
        dispNames = engNames

    transformResNet = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    model = tvModels.resnet101()

    model.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=model.fc.in_features,
                out_features=240
            ),
            torch.nn.Sigmoid()
        )

    mState = torch.load(f"{mPath}", weights_only=True)

    model.load_state_dict(mState['model_state_dict'])
    print("Loaded model!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    outputs = []
    for (root, dirs, file) in os.walk(folderName):
        for f in file:
            if "annotated" in f:
                continue
            try:
                oriimage = cv2.imread(os.path.join(root,f))
                image = cv2.cvtColor(oriimage, cv2.COLOR_BGR2RGB)
                image = transformResNet(image).cuda().unsqueeze(0)
                output = pd.Series(model(image).data.cpu().numpy().flatten())
                idx = output.nlargest(5).index.values.tolist()
                descr = f"{f}\n"
                y0, dy = 5, 13
                for i in idx:
                    descr = descr + f"{dispNames[i]}: {output[i]*100:.1f}%\n"
                print(descr)
                imResize = cv2.resize(oriimage, (512,512))
                for i, line in enumerate(descr.split('\n')):
                    y = y0 + i*dy
                    # imResize = cv2.putText(imResize, line, (1,y), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color = (0,255,255))
                    draw_text(imResize,line,pos=(1,y),font_thickness=1)

                cv2.imwrite(os.path.join(root,f"annotated_{f}"),imResize)

            except Exception as e:
                print(e)
                print(f"Couldn't load image {f}")

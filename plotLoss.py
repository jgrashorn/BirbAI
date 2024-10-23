import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

mName = "model_temp_RN101"
filename = f"{mName}.log"
data = pd.read_csv(filename)

Nconv = 20

epochsTraining = data["epoch"].to_numpy()
tLoss = data["loss"].to_numpy()
epochsValidation = data["epoch"][~np.isnan(data["validation loss"].to_numpy())].to_numpy()
vLoss = data["validation loss"][~np.isnan(data["validation loss"].to_numpy())].to_numpy()

lrIdx = int(np.floor(len(epochsTraining)/5))

m1, b1 = np.polyfit(epochsTraining[:lrIdx],tLoss[:lrIdx],1)
m2, b2 = np.polyfit(epochsTraining[len(epochsTraining)-lrIdx:],tLoss[len(epochsTraining)-lrIdx:],1)

lrFunction = lambda x, m, b: m*x + b

lrLower = lrFunction(epochsTraining[:int(np.floor(2*len(epochsTraining)/3))], m1, b1)
lrUpper = lrFunction(epochsTraining[int(np.floor(1*len(epochsTraining)/3)):], m2, b2)

mavShortTraining = np.convolve(tLoss,np.ones(Nconv)/Nconv,mode='valid')
mavLongTraining = np.convolve(tLoss,np.ones(Nconv*10)/(Nconv*10),mode='valid')

trainShortLength = np.floor(len(epochsTraining)-len(mavShortTraining))
trainLongLength = np.floor(len(epochsTraining)-len(mavLongTraining))

idxTrainShort = [int(np.floor(trainShortLength/2)), int(len(epochsTraining)-np.floor(trainShortLength/2))]
idxTrainLong = [int(np.floor(trainLongLength/2)), int(len(epochsTraining)-np.floor(trainLongLength/2))]

if np.diff(idxTrainShort) != len(mavShortTraining):
    idxTrainShort[0] += 1
if np.diff(idxTrainLong) != len(mavLongTraining):
    idxTrainLong[0] += 1

fig, ax1 = plt.subplots(figsize=(12,8))

color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training loss', color=color)
ax1.plot(epochsTraining,tLoss, 'x', color=color)
ax1.plot(epochsTraining[idxTrainShort[0]:idxTrainShort[1]],mavShortTraining,'r--')
ax1.plot(epochsTraining[idxTrainLong[0]:idxTrainLong[1]],mavLongTraining,'r--')

ax1.plot(epochsTraining[:int(np.floor(2*len(epochsTraining)/3))],lrLower,'k--',linewidth=3)
ax1.plot(epochsTraining[int(np.floor(1*len(epochsTraining)/3)):],lrUpper,'k--',linewidth=3)
ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Validation loss [%]')
ax2.plot(data["epoch"][~np.isnan(data["validation loss"].to_numpy())].to_numpy(),data["validation loss"][~np.isnan(data["validation loss"].to_numpy())].to_numpy(),color=color)

fig.tight_layout()
fig.suptitle(f"Remaining epochs: {-b2/m2:.0f}")
plt.show()
fig.savefig(f'LossPlots/{mName}_LossEpoch{epochsTraining[-1]}.png')
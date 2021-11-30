import torch
import argparse
from dataset import createTrainLoader
from resnet import createResnetArchitecture
from metric import compareActivation
import matplotlib.pyplot as plt

"""
Similarity file provides arguments for the 
creating the heatmaps between the model layers.

"""

parser = argparse.ArgumentParser(description='Custom ResNet Architecture')
parser.add_argument('--datapath', default='.', type=str)
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--training_size', default=1, type=float)
parser.add_argument('--kMini', default=1, type=int)
userOpt = parser.parse_args()

# creating testLoader
testLoader = createTrainLoader(userOpt, False)

# creating model function for forward pass internals
model, params = createResnetArchitecture(userOpt.depth, userOpt.width, 10, flag=1)

# loading params from the stored trained model on specified depth and width
params = {}
params = torch.load(userOpt.datapath)
print("Parameters Loaded...")

# loading the test data -- we are taking only one batch due to limited computational resources
for i,batch in enumerate(testLoader):
  data, target = batch
  data = data.cuda()
  break 

print("Storing the internal representation for batch...")

out, storeComputations = model(data, params=params, mode=True)

print("Computations stored")

print("Comparing activations...")
sim = compareActivation(storeComputations)

print("CKA score calculated...")

# plotting figures 
plt.figure(figsize=(5, 5),dpi=50)
axes = plt.imshow(sim, cmap='magma', vmin=0.0,vmax=1.0)
nameTitle = "ResNet-"+str(userOpt.depth)+" "+str(userOpt.width)+"x" + ' ' + str(userOpt.training_size)
plt.title(nameTitle)
plt.xlabel("Layer")
plt.ylabel("Layer")
axes.axes.invert_yaxis()
pathSaveFigName = "./ResNet-"+str(userOpt.depth)+" "+str(userOpt.width)+"x" + ' ' + str(userOpt.training_size) + "comparison.png"
plt.savefig(pathSaveFigName, dpi=400)


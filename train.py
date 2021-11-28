import argparse
import torch
import torch.utils.data
from torch.optim import SGD
from torch.backends import cudnn
from dataset import createTrainLoader
from resnet import createResnetArchitecture

"""
    This file trains our custom resnet model
"""

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Custom ResNet Architecture')
parser.add_argument('--depth', default=14, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--training_size', default=1, type=float)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=5, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')

userOpt = parser.parse_args()

# creating train loader 
trainLoader = createTrainLoader(userOpt, True)
num_classes = 10 if userOpt.dataset == 'CIFAR10' else 100
model, params = createResnetArchitecture(userOpt.depth, userOpt.width, num_classes)
print('parsed options:', vars(userOpt))

torch.manual_seed(userOpt.seed)

# creating optimizer
def createOptimizer(opt, lr):
        return SGD([val for val in params.values() if val.requires_grad], lr, momentum=0.9, weight_decay=opt.weight_decay)

optimizer = createOptimizer(userOpt, userOpt.lr)

epoch = userOpt.epochs

# training loop
for j in range(epoch):
    trainLoss = 0
    for i, (input, target) in enumerate((trainLoader)/userOpt.training_size):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        
        output = model(input_var, params, True)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainLoss += loss.item()

        if i % 50 == 0:
            print('Epoch:', j , 'Data:', i)
            # print(prec1)
    print("Epoch:", j, "Loss",trainLoss/len(trainLoader)/userOpt.training_size)

print('Model Training done...')
# storing model
PATH = './ResnetParamStore_'+ userOpt.depth + '_' + userOpt.width + '.pth'
torch.save(params, PATH)

print('Model Params stored')


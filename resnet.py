import torch
import torch.nn.functional as tFunc
from initialization import *

def createResnetArchitecture(depth, filterK, numClasses, flag=0):
    """
    Create the Resnet Architecture based on required depth, width and number of classes
    """
    assert (depth-2)%6==0, "depth should be of the form 6n+2" # our experiments involve depths 26,38,44,56,110,164 
    layers = depth//6
    widths = [width*filterK for width in (16,32,64)]
    storeComputation = []

    def generateBlockParameters(inLayer, outLayer):
        """
        Create parameters required for each block structure based on input and output sizes
        """
        params = {'conv0': convolutionalParamsInitialization(inLayer, outLayer, 3),
            'conv1': convolutionalParamsInitialization(outLayer, outLayer, 3),
            'bn0': batchNormalizationParamsInitialization(inLayer),
            'bn1': batchNormalizationParamsInitialization(outLayer),
            'convdim': convolutionalParamsInitialization(inLayer, outLayer, 1) if inLayer != outLayer else None}
        return params
    
    def generateGroupParameters(inp, out, count):
        """
        Create parameters required for each block structure in the group based on input, output and group size
        """
        params = {'block%d' %i:generateBlockParameters(inp if i==0 else out, out) for i in range(count)}
        return params

    flatParameters = castParams(flattenParams({
        'conv0': convolutionalParamsInitialization(3, 16, 3),
        'group0': generateGroupParameters(16, widths[0], layers),
        'group1': generateGroupParameters(widths[0], widths[1], layers),
        'group2': generateGroupParameters(widths[1], widths[2], layers),
        'bn': batchNormalizationParamsInitialization(widths[2]),
        'fc': linearParamsInitialization(widths[2], numClasses),
    }))

    setRequiresGradExceptBatchNormalization(flatParameters)

    def blockStructure(inLayerX, params, base, mode, stride):
        """
        Create the resnet block structure using specified parameters 
        """
        # TODO
        out1bn = batchNormalization(inLayerX, params, base + '.bn0', mode)
        out1 = tFunc.relu(out1bn, inplace=True)
        yDoubt = tFunc.conv2d(out1, params[base + '.conv0'], stride=stride, padding=1)
        out2bn = batchNormalization(yDoubt, params, base + '.bn1', mode)
        out2 = tFunc.relu(out2bn, inplace=True)
        zDoubt = tFunc.conv2d(out2, params[base + '.conv1'], stride=1, padding=1)
        
        if(flag):
            storeComputation.append(out1bn)
            storeComputation.append(out1)
            storeComputation.append(yDoubt)
            storeComputation.append(out2bn)
            storeComputation.append(out2)
            storeComputation.append(zDoubt)
    
        res = zDoubt
        
        if (base + '.convdim' in params):
            res += tFunc.conv2d(out1, params[base + '.convdim'], stride=stride)
            if flag:
                storeComputation.append(res)
        else:
            res += inLayerX
            if flag:
                storeComputation.append(res)
        return res
    
    def groupStructure(out, params, base, mode, stride):
        """
        Create the resnet group structure using specified parameters and blocks
        """
        for i in range(layers):
            out = blockStructure(out, params, '%s.block%d' % (base,i), mode, stride if i==0 else 1)
        return out

    def forwardPass(input, params, mode):
        """
        forward pass internals
        """
        inLayerX = tFunc.conv2d(input, params['conv0'], padding=1)
        if(flag==1):
            storeComputation.append(inLayerX)
        gp0 = groupStructure(inLayerX, params, 'group0', mode,  1)
        gp1 = groupStructure(gp0, params, 'group1', mode, 2)
        gp2 = groupStructure(gp1, params, 'group2', mode, 2)
        outbn = batchNormalization(gp2, params, 'bn', mode)
        out = tFunc.relu(outbn)
        if flag:
                storeComputation.append(outbn)
                storeComputation.append(out)
        out = tFunc.avg_pool2d(out, 8, 1, 0)
        if flag:
            storeComputation.append(out)
        out = out.view(out.size(0), -1)
        out = tFunc.linear(out, params['fc.weight'], params['fc.bias'])
        if flag:
                storeComputation.append(out)
        if(flag==1):
            return out, storeComputation
        return out
    
    return forwardPass, flatParameters



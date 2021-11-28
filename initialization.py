import torch
import torch.nn.functional as tFunc
from torch.nn.init import kaiming_normal_
from nested_dict import nested_dict

def castParams(parameters, dtype='float'):
    """ Cast the params

    Args:
        parameters ([type]): parameters 
        dtype (str, optional): type for params. Defaults to 'float'.
    """
    if isinstance(parameters, dict):
        return {keyI: castParams(valueI, dtype) for keyI,valueI in parameters.items()}
    else:
        return getattr(parameters.cuda() if torch.cuda.is_available() else parameters, dtype)()


def convolutionalParamsInitialization(inLayer, outLayer, filterK=1):
    """ tensor intializtion of convParams 

    Args:
        inLayer ([type]): count of input layer
        outLayer ([type]): count of output layer
        filterK (int, optional): filter lenght/width. Defaults to 1.
    """
    return kaiming_normal_(torch.Tensor(outLayer, inLayer, filterK, filterK))


def linearParamsInitialization(inLayer, outLayer):
    """ tensor intializtion of linearParams 

    Args:
        inLayer ([type]):  count of input layer
        outLayer ([type]):  count of output layer
    """
    return {'weight': kaiming_normal_(torch.Tensor(outLayer, inLayer)), 'bias': torch.zeros(outLayer)} 

def batchNormalizationParamsInitialization(paraN):
    """ tensor initialization of batchNormParams

    Args:
        paraN ([type]): number of parasN for intialization
    """
    return {'weight': torch.rand(paraN),
            'bias': torch.zeros(paraN),
            'running_mean': torch.zeros(paraN),
            'running_var': torch.ones(paraN)}


def flattenParams(parameters):
    """ flattens tehe parameteres in the nested directory

    Args:
        parameters ([type]): parameters
    """
    return {'.'.join(keyI): valueI for keyI, valueI in nested_dict(parameters).items_flat() if valueI is not None}


def batchNormalization(inLayerX, parameters, base, mode):
    """ batchNormalizationLayer

    Args:
        inLayerX ([type]): input layer 
        parameters ([type]): parameters
        base ([type]): base 
        mode ([type]): mode of training

    Returns:
        [type]: [description]
    """
    return tFunc.batch_norm(inLayerX, weight=parameters[base + '.weight'], bias=parameters[base + '.bias'], 
                        running_mean=parameters[base + '.running_mean'], running_var=parameters[base + '.running_var'], 
                        training=mode)


def setRequiresGradExceptBatchNormalization(params):
    """ for each params set the requires_grad as true except the batchNormalization layer
    
    Args:
        params ([type]): parameters
    
    """ 
    
    for keyI, valueI in params.items():
        if not keyI.endswith('running_mean') and not keyI.endswith('running_var'):
            valueI.requires_grad = True

import numpy as np
import tqdm

def unbiasedMetricHSIC(KMatrix, LMatrix):
    """Computes an unbiased estimator of HISC. 
       This is equaltion 2 from the paper.

    Args:
        KMarix ([type]): [description]
        LMatrix ([type]): [description]
    """
    nKShape = KMatrix.shape[0] # shape of KMatrix
    unitVector = np.ones(shape=(nKShape)) # creating unit vector with nKShape
    
    np.fill_diagonal(KMatrix, val=0) # KTilde (as in paper) # filling the diagonal entries with 0
    np.fill_diagonal(LMatrix, val=0) # LTilde (as in paper) # filling the diagonal entries with 0 
    
    # first part helper intermediaries
    traceFirstPart = np.trace(np.dot(KMatrix, LMatrix)) 
    
    # second part helper intermediaries
    numerator1SecondPart = np.dot(np.dot(unitVector.T, KMatrix), unitVector)
    numerator2SecondPart = np.dot(np.dot(unitVector.T, LMatrix), unitVector)
    denominatorSecondPart = (nKShape-1)*(nKShape-2)
    secondPartMerged = np.dot(numerator1SecondPart, numerator2SecondPart) / denominatorSecondPart
    
    # third part helper intermediaries
    multiplier1ThirdPart = 2/(nKShape-2)
    multiplier2ThirdPart = np.dot(np.dot(unitVector.T, KMatrix), np.dot(LMatrix, unitVector))
    thirdPartMerged = multiplier1ThirdPart*multiplier2ThirdPart
    
    # COMPLETE EQUATION
    unbiasedHSICEquation = 1/(nKShape*(nKShape-3)) * (traceFirstPart + secondPartMerged - thirdPartMerged)
    
    return unbiasedHSICEquation
    

def centeredKernelAlignmentMetricCKA(XMatrix, YMatrix):
    """ CKA metric - Computes CKA between two matrices. 
        This is equation 1 from the paper

    Args:
        XMatrix ([type]): [description]
        YMatrix ([type]): [description]
    """
    numeratorInCKA = unbiasedMetricHSIC(np.dot(XMatrix, XMatrix.T), np.dot(YMatrix, YMatrix.T))
    denominator1InCKA = unbiasedMetricHSIC(np.dot(XMatrix, XMatrix.T), np.dot(XMatrix, XMatrix.T))
    denominator2InCKA = unbiasedMetricHSIC(np.dot(YMatrix, YMatrix.T), np.dot(YMatrix, YMatrix.T))
    
    # final CKA computation
    equationCKA = numeratorInCKA / np.sqrt(denominator1InCKA*denominator2InCKA) 
    
    return equationCKA


def calculatingCKAScore(activationAMatrix, activationBMatrix):
    """ Computes the linear CKA to measure their similarity

    Args:
        activationAMatrix ([type]): [description]
        activationBMatrix ([type]): [description]
    """
    # unfolding the activations
    shapeAMatrix = activationAMatrix.shape 
    activationAMatrix = np.reshape(activationAMatrix, newshape=(shapeAMatrix[0], np.prod(shapeAMatrix[1:])))
    
    shapeBMatrix = activationBMatrix.shape
    activationBMatrix = np.reshape(activationBMatrix, newshape=(shapeBMatrix[0], np.prod(shapeBMatrix[1:])))
    
    # compute CKA score
    ckaScore = centeredKernelAlignmentMetricCKA(activationAMatrix, activationBMatrix)
    
    # deleting after computing the score
    del activationAMatrix 
    del activationBMatrix
    
    return ckaScore

def compareActivation(storedValArr, userOpt):
    """ This function comapares the activation 

    Args:
        storedValArr ([type]): [description]
        userOpt ([type]): [description]

    """
    # print(len(storedValArr))
    # finalArr = np.zeros(shape=(len(storedValArr), len(storedValArr)))
    # for i in range(len(storedValArr)):
    #     for j in range(len(storedValArr)):
    #         ckaScore = calculatingCKAScore(storedValArr[i].detach().cpu().numpy(), storedValArr[j].detach().cpu().numpy())
    #         finalArr[i, j] = ckaScore 
    # return finalArr
    print(len(storedValArr))
    result_array = np.zeros(shape=(len(storedValArr), len(storedValArr)))
    i = 0
    for outputA in tqdm.tqdm_notebook(storedValArr):
        j = 0
        for outputB in tqdm.tqdm_notebook(storedValArr):
            cka_score = calculatingCKAScore(outputA, outputB)
            result_array[i, j] = cka_score
            j+=1
        i+= 1

    return result_array
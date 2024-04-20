# References for the below modules that are imported:
# For json: [61fceb]; pathlib: [889500]; chainer: [170ecf]; matplotlib:
# [7214f3]
####################################################################################################
#                                                                                                  #

import json
import pathlib
import chainer
import matplotlib.pyplot

#                                                                                                  #
####################################################################################################

import networks



# Reads a dataset created by sampler.py
def shapeDataset(datasetFolder):
   handle  = open(pathlib.Path(datasetFolder, "truths.json"))
   d       = json.load(handle)
   handle.close()
   order   = ["tri", "sq"]
   paths   = [   (p,  [ d[p][s] for s in order if s in d[p] ])   for p in d]
   return chainer.datasets.LabeledImageDataset(paths,
                                               datasetFolder,
                                               label_dtype=chainer.numpy.float32)

# A function meant to be used for both testing and validation. Returns the mean of the absolute
# error in counts (per shape type). Works with the dataset produced by sampler.py. "network" refers
# to the loaded neural network, while "dataset" is an instance of
# LabeledImageDataset[3e890c]. As suggested by David Crandall[7674ac], we
# plot the number of shapes guessed against the number of shapes in the image.
def performance(network, dataset):
    
    # Next two variables are for graphing
    allPredictions  = []
    allGoals        = []

    with chainer.no_backprop_mode():

        iterator = chainer.iterators.SerialIterator(dataset, 30, shuffle=False, repeat=False)
        summed   = chainer.Variable(
                        chainer.numpy.zeros(
                            network.countableShapes,
                            dtype=chainer.numpy.float32
                        )
                   )
        summed.to_device(network.device)
        for batch in iterator:
            batch   = chainer.dataset.concat_examples(batch, network.device)
            out     = network(batch[0])
            # We are using the value that "correct" is set to for graphing, resulting in it being
            # used in two places, so that is why we have this single variable (for reuse)
            correct = batch[1]
            error   = chainer.functions.absolute_error(out, correct)
            summed += chainer.functions.sum(error, 0)
            allPredictions  += out.data.tolist()
            allGoals        += correct.tolist()

        
        matplotlib.pyplot.plot(allGoals, allPredictions, ".k")
        matplotlib.pyplot.savefig("graph.svg")
        averageError = summed / len(dataset)
        return averageError

# This function allows the easy changing of the kinds of networks we are using (see README,
# networks.py). Sigmoid and sine are available because the second derivative may be larger (or, in
# the opposite direction more negative), which hopefully makes localization.py work (as opposed to
# SoftPlus[cf158f] or ModifiedSoftPlus, whose derivatives is pretty flat most of the time).
# Also, there likely are no overflow issues (as discussed in networks.Convolutional's comment) with
# either sine or sigmoid.
def getEmptyModel(network, activation, base, renormalize, predictionSize):

    f = None
    if activation == "pureChainerMSP":
        f = networks.pureChainerMSP
    if activation == "modifiedSoftPlus":
        f = networks.modifiedSoftPlus
    if activation == "sin":
        f = chainer.functions.sin
    if activation == "relu": # Using ReLU [bffa08]
        f = chainer.functions.relu
    if activation == "sigmoid":
        f = chainer.functions.sigmoid
    if activation == "softplus": # SoftPlus[cf158f]
        f = chainer.functions.softplus

    model = None
    if network == "convolutional":
        model = networks.Convolutional(predictionSize, f, base, renormalize)
    if network == "convolutionallarge":
        model = networks.ConvolutionalLarge(predictionSize, f, base, renormalize)
    
    return model

# Loads the network whose class is in networks.py and is trained by learn.py and returns it
def recreateNetwork(networkPath, settingsPath):
    model         = None
    modelFilename = pathlib.Path(networkPath).parts[-1]
    
    handle        = open(settingsPath)
    settings      = json.load(handle)
    handle.close()

    model = getEmptyModel(settings["model"],
                          settings["activation"],
                          settings["base"],
                          settings["renormalize"],
                          settings["howManyShapeTypes"])   
    chainer.serializers.load_npz(networkPath, model)
    
    return model
        
        
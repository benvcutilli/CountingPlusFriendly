# Package from [170ecf]
import chainer

import networks
import networkcomponents
import tackons
import attacks




# Including the ability from chainer.functions.sum(...)[1b144c] and functions/methods in
# PyTorch [05c8ff] as well (for example, [7a1513]/[4d49cb]), to
# select just the dimensions (the "dimensions" parameter) that will include its data in a single
# resultant norm. I actually think that I was expecting something like PyTorch's
# torch.norm(...)[7a1513] and/or torch.Tensor.norm(...)[4d49cb] to be
# in Chainer, but it doesn't exist, so I wrote a function that computes it.
#   "dimensions" takes its format from "axis" from [1b144c], a tuple of dimension indices
# (although "axis" accepts other things as well, which this function doesn't). Also accepts None
# so that None can be specified when calling chainer.functions.sum(...)
def n_norm(n, data, dimensions):
    
    return chainer.functions.sum(
                chainer.functions.absolute(data) ** n,
                dimensions,
                True
           ) ** (1/n)


# † Dynamically loads callables from modules as that makes trying out new features easily
def getFromModules(thing, parent, fallback):

    if thing == "none":
        return None

    item = getattr(parent, thing, None)
    if item == None:
        item = getattr(fallback, thing)
    
    return item 



# This function does basically the same thing that "Counting"'s common.getEmptyModel(...) does (in
# that it creates a fresh network using differing attributes. † This fulfills the pick-and-choose
# mentality of this codebase. Need datasetHeight for networks.SzegedyLinear because, as outlined in
# that class's comment, it is actually a convolutional network and these networks allow different
# datasets to be used. "commandLine" is the data from meta.json, as is discussed in workbench.py.
# As in Counting (though the file name is settings.json), meta.json is meant to be read by the
# calling script and then passed right to this function
def create(commandLine,
           datasetDepth,
           datasetWidth,
           datasetHeight):


    layers = {
        "Convolution2D":      chainer.links.Convolution2D,
        "PairwiseDifference": networkcomponents.PairwiseDifference,
        "Angular":            networkcomponents.Angular
    }
    layerClass      = layers[commandLine["layer"]]


    
    activationClassName = commandLine["activation"]
    if activationClassName == "donothing":
        # This idea is from the networks found in Counting/networks.py where we don't want to use
        # batch renormalization at all (so we pass in the "identity" function); however, we are
        # doing this instead for the activation function of the network.
        activationClass = lambda passOn: passOn
    else:
        activationClass = getFromModules(activationClassName, networkcomponents, chainer.functions)



    if commandLine["network"] == "SzegedyLinear":
        # removed multiplication of datasetHeight by datasetWidth and datasetDepth for the second
        # parameter because of the number of input dimensions being changed to constant to ease
        # black-box testing
        return networks.SzegedyLinear(datasetDepth,
                                      datasetHeight,
                                      layerClass,
                                      activationClass,
                                      commandLine["renormalize"],
                                      commandLine["szegedy_linear_modifier"])
    elif commandLine["network"] == "MalleableConvolutional":
        
        return networks.MalleableConvolutional(datasetDepth, layerClass, activationClass)








# † An object that computes all the specified losses that the user specifies
class ArbitraryLoss(object):

    def __init__(self, standardLossArgs, extraLossArgs):
        super().__init__()
        self.standardLossArgs = standardLossArgs
        self.extraLossArgs    = extraLossArgs

    # Argument rationales can be read above networks.ArbitraryLossClassifier. "aggregate" was made
    # to do the same thing (and have the same default value) as the parameter with the same name in
    # function(s) found in tackons.py
    def __call__(self, toCompare, toClassify, predictWith, aggregate=True):
        
        output = predictWith(toClassify)

        misprediction    = []
        for l in self.standardLossArgs:
            lFunction = getFromModules(l["function"], tackons, chainer.functions)
            # † Intended to be used with functions with signatures like
            # [cf684f], possibly done because
            # chainer.Classifier[e10c70] does it.
            misprediction.append( lFunction(output, toCompare) * l["weight"] )
        
        # This is pre-calculated for efficiency reasons
        totalMispredictionLoss = chainer.functions.sum(
                                    chainer.functions.vstack(misprediction)
                                 )

        
        nonMisprediction = []
        if self.extraLossArgs:
            for l in self.extraLossArgs:
                lFunction = getFromModules(l["function"], tackons, chainer.functions)
                # † Assumes a function signature that tackons.jacobianNorm and tackons.hessianNorm
                # has
                nonMisprediction.append(
                    lFunction(
                        toClassify,
                        totalMispredictionLoss
                    ) * l["weight"]
                )



        final = (chainer.functions.sum(
                    chainer.functions.vstack(nonMisprediction)
                ) if self.extraLossArgs != None else 0)                                            \
                                    +                                                              \
                         totalMispredictionLoss
        
        chainer.report(
            {
                "Training Error (class)": totalMispredictionLoss,
                "Training Error": final
            }
        )
        
        
        return final











def oneHotConverter(samples, to):

    to                    = chainer.get_device(to)

    firstStep             = chainer.dataset.convert.concat_examples_func(samples, to)

    # The length of the second dimension of this tensor is set to the number of classes in
    # MNIST^^^mnist and CIFAR-10[c7fedb]
    secondStep            = to.xp.zeros(
                                (firstStep[1].shape[0], 10),
                                dtype=to.xp.float32
                            )
    firstDimensionIndices = chainer.numpy.array(
                                range(firstStep[1].shape[0]),
                                dtype=to.xp.int32
                            )
    secondStep[firstDimensionIndices, firstStep[1]] = 1.0
    
    return to.send(   (firstStep[0], secondStep)   )


# This creates adversarial examples to use for training as described in [7e5d83]
class AdversarialTrainingConverter(chainer.dataset.Converter):

    def __init__(self, network, loss, norm, attack):
        super().__init__()

        self.network = network
        self.loss    = loss
        self.norm    = norm
        self.attack  = attack
    
    def __call__(self, batch, memoryOf):
        
        images, truths = oneHotConverter(batch, memoryOf)
        if self.attack == attacks.FGSM:
            images = attack(images, truths, self.network, self.loss, self.norm)
        else:
            # 100 for the number of times FGSM is run, only using one randomized point, and the
            # formula passed in for attack.PGD's "step" parameter are the same as that in
            # [7e5d83] except that, for "step", we round as we are assuming integer-valued
            # pixels. The aforementioned formula is in section 5 of that paper.
            images = self.attack(chainer.Variable(images),
                                 chainer.Variable(truths),
                                 self.network,
                                 self.loss,
                                 round(2.5 * (self.norm/100)),
                                 1,
                                 100,
                                 self.norm)
        
        return (images, truths)






# These values will be used to generate the correct widths for networks.SzegedyLinear (as the input
# needs to be (batch, "channels", 1, "width" * "height"); see above networks.SzegedyLinear for why),
# and will also be used to tell networks.MalleableConvolutional how to shape the first layer
# filters. The values of the "trainOn" argument of workbench.py were chosen for these keys.
imageMetadata = {
    "mnistinverted": { "channels": 1, "width": 28, "height": 28 },
    "mnist":         { "channels": 1, "width": 28, "height": 28 },
    "cifar10":       { "channels": 3, "width": 32, "height": 32 },
}
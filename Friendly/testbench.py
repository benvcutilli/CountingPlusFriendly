# † Each argument for this program can swap in a different piece to try attacks in different ways,
# different models, etc. Can test on MNIST^^^mnist^^^ and eventually CIFAR-10^^^cifar^^^, the
# testing or training subsets. When adversarial sample creation is activated, the whole dataset is
# attacked as that is what is done in ^^^intriguingproperties^^^; however there is an ability (well,
# not as of this writing) to do one image at a time to be able to do image analysis as is also done
# in that paper.

# From ^^^pythonargparse^^^ (the package is, not this line of code)
import argparse

# Package from ^^^json^^^
import json

# ^^^chainer^^^, the reference for this package
import chainer

# Pathlib^^^pythonpathlib^^^
import pathlib

import attacks
import common
import tackons
import networks

chainer.global_config.train = True

argumentParser = argparse.ArgumentParser()
argumentParser.add_argument("testAgainst",
                            type=str,
                            help="Same meaning as in workbench.py (with the same name), except \
                                 that we are reading from the folder to get meta.json and the \
                                 network")
argumentParser.add_argument("generateAgainst",
                            type=str,
                            help="The network used to generated adversarial examples. This is \
                                 useful for black box attacks")
# Choosing between MNIST's^^^mnist^^^ (pixel values for MNIST are flipped as described in the
# README) and CIFAR-10's^^^cifar^^^ training and testing subsets
argumentParser.add_argument("input",
                            type=str,
                            choices=["MNISTtest", "MNISTtrain", "CIFARtest", "CIFARtrain"],
                            help="The dataset to attack")
# † The options here are the same as the names of the attacks in attacks.py so that no code other than
# the attack itself needs to be written, easing the process of implementing new attacks to benchmark
argumentParser.add_argument("--attack",
                            type=str,
                            choices=["FGSM", "PGD"],
                            default="empty",
                            help="Allows you to pick your desired attack if you would like to \
                                 attack instead of doing normal evaluation")

# Regarding ^^^explaining^^^'s FGSM and ^^^towards^^^'s PGD
argumentParser.add_argument("--step",
                            type=float,
                            help="The step size (0 to 255 range) you want to be using during FGSM (\
                                 https://arxiv.org/pdf/1412.6572.pdf) or PGD \
                                 (https://arxiv.org/pdf/1706.06083.pdf)",
                            default=2)

# The default amount of noise is the rounded 0-255 equivalent of what ^^^towards^^^ uses, so for
# comparison's sake it is the default here
argumentParser.add_argument("--allowed-norm",
                            type=int,
                            help="How much an adversarial pixel value can differ from the original \
                                 example (a.k.a the L-infinity norm)",
                            default=77)

# Configuring PGD^^^towards^^^. Defaults are based on the optimal settings for PGD found in table 1
# of ^^^towards^^^
####################################################################################################
#                                                                                                  #

argumentParser.add_argument("--num-randoms-pgd",
                            type=int,
                            help="How many times PGD should optimize within the L-infinity ball",
                            default=20)
argumentParser.add_argument("--num-steps-pgd",
                            type=int,
                            help="Number of times to update an adversarial example during \
                                  optimization via PGD",
                            default=100)

#                                                                                                  #
####################################################################################################

parameters = argumentParser.parse_args()

attackRoot = pathlib.Path(parameters.generateAgainst)
victimRoot = pathlib.Path(parameters.testAgainst)

# Just as in Counting/test.py, we are reading the file containing all the command line arguments
# passed to the training Python script, and using that information to make the network (and maybe
# other things I have forgotten to list here)
####################################################################################################
#                                                                                                  #

attackNetworkMeta      = json.loads( (attackRoot / "meta.json").read_text() )
victimNetworkMeta      = json.loads( (victimRoot / "meta.json").read_text() )
datasetCharacteristics = common.imageMetadata[attackNetworkMeta["trainOn"]]
attackNetwork          = common.create(
                            attackNetworkMeta,
                            datasetCharacteristics["channels"],
                            datasetCharacteristics["width"],
                            datasetCharacteristics["height"],
                         )
victimNetwork          = common.create(
                            victimNetworkMeta,
                            datasetCharacteristics["channels"],
                            datasetCharacteristics["width"],
                            datasetCharacteristics["height"],
                         )
chainer.serializers.load_npz( attackRoot / "finalnetwork.npz", attackNetwork )
chainer.serializers.load_npz( victimRoot / "finalnetwork.npz", victimNetwork )
attackNetwork.to_device("@cupy:0")
victimNetwork.to_device("@cupy:0")

#                                                                                                  #
####################################################################################################


# Sticking with three dimensions per sample no matter the network so that way we can do black-box
# attacks without switching between one dimension and three (also, relatedly but inadvertently, this
# is made possible by networks.SzegedyLinear being a linear network that is emulated with
# convolutional layers). Used to be code here to figure out which network needed what
# dimensionality.
dataset         = chainer.datasets.get_mnist(
                        ndim=3,
                        scale=255
                  )[0 if parameters.input == "MNISTtrain" else 1]
dataset.repeat  = False



# † Getting the desired attack and loss function dynamically (as in there is no attack-specific code
# to write in any other files, at least for attacks that don't require options for additional
# settings)
attack          = getattr(attacks, parameters.attack, None)
# None is passed in here as parse_args(...)^^^argumentparserparseargs^^^ does so when the user
# doesn't pass in a value for --other-loss-term; --other-loss-term can be found in workbench.py
lossFn          = common.ArbitraryLoss(attackNetworkMeta["classification_loss_term"], None)







# Now testing the network against the attack (or no attack)
####################################################################################################
#                                                                                                  #

chainer.global_config.enable_backprop = False

seenFirstBatch = False
predictions    = None
labels         = None
# Using an iterator to create batches to send to the GPU. Making an iterator for this purpose, as
# well as setting the third parameter to true, is a strategy from Counting/test.py.
for pairs in chainer.iterators.SerialIterator(dataset, 200, False, False):

    # Attacking based on what was requested
    ################################################################################################
    #                                                                                              #

    # Mimicking what an Updater probably does^^^chainerupdater^^^, passing in the pair from the
    # dataset to a converter^^^chainerconverter^^^ with the pair being encapsulated in another
    # list/tuple that would hold each pair if the batch contained more than one pair. Need to do
    # this twice, with oneHotConverter providing us with a one-hot encoding of the ground truth for
    # use in attacking images.
    converted       = common.oneHotConverter(pairs, "@cupy:0")
    oneHotTruths    = chainer.Variable(  converted[1]  )
    batchFormImages = chainer.Variable(  converted[0]  )
    batchFormTruths = chainer.dataset.concat_examples(pairs, "@cupy:0")[1]


    predictForThis = batchFormImages
    if parameters.attack != None:
        chainer.global_config.enable_backprop = True
        if attack == attacks.FGSM:
            predictForThis  = attack(batchFormImages,
                                     oneHotTruths,
                                     attackNetwork,
                                     lossFn,
                                     parameters.step)
        if attack == attacks.PGD:
            predictForThis  = attack(batchFormImages,
                                     oneHotTruths,
                                     attackNetwork,
                                     lossFn,
                                     parameters.step,
                                     parameters.num_randoms_pgd,
                                     parameters.num_steps_pgd,
                                     parameters.allowed_norm)
        chainer.global_config.enable_backprop = False
    
    # Making one giant batch from the output of the network and the labels for that batch for the
    # whole dataset.
    if seenFirstBatch == False:
        predictions    = victimNetwork(predictForThis)
        labels         = batchFormTruths
        seenFirstBatch = True
    else:
        predictions    = chainer.functions.concat( (predictions, victimNetwork(predictForThis)), 0)
        labels         = chainer.functions.concat( (labels, batchFormTruths),              0)
        

    #                                                                                              #
    ################################################################################################

metric = chainer.functions.accuracy(  predictions,  labels  )
print(   "Fraction correct: " + str(metric)   )

#                                                                                                  #
####################################################################################################

# argparse package: ^^^pythonargparse^^^
import argparse

# The package supplied by ^^^chainer^^^
import chainer

# Package from ^^^pythonpathlib^^^
import pathlib

# The ^^^pythonjson^^^ package
import json

# See ^^^pythonre^^^ for this library
import re

# Importing ^^^pillow^^^
import PIL

import common
import tackons
import networks
import RMaxDMaxModifier
import attacks


def accuracyWrapper(inputs, labels):
    fractionRight = chainer.functions.accuracy(inputs, labels)
    # Doing the same thing as "Training Error" and "Training Error (class)" below and not assigning
    # this observation an observer
    chainer.report({"Performance": fractionRight})

    return fractionRight





argumentParser = argparse.ArgumentParser()

# † "--classification-loss-term", "--other-loss-term", "network", "trainOn", "--algorithm",
# "--renormalize", "--network-modifier", "--activation", "--perturbation-norm", and
# "--adversarial-training-with" are here so that you can be modular in how you train your network
# (including the weights discussed for "--other-loss-term" and "classification-loss-term"). The idea
# of having the "--renormalize", "--device", "trainingProducts", "--learning-rate", "--epochs", and
# "--notes" options is from the "Counting" subproject (sibiling to this one, "Friendly");
# specifically, these are akin to the "--renormalize", "--device", "saveFolder", "--learning-rate",
# "--epochs", and "--notes" options, respectively. In addition, the option names for them (except
# trainingProducts) are from there, as well as how they function (including trainingProducts);
# however, in this file, "--epochs" performs early stopping instead of just counting the number of
# epochs. See Counting/learn.py for comments for its equivalent parameters that aren't mentioned
# here.
####################################################################################################
#                                                                                                  #

argumentParser.add_argument("network",
                            type=str,
                            choices=["SzegedyLinear", "MalleableConvolutional"],
                            help="Which network you want to use. Names are the network names seen  \
                                  in networks.py")

# The full reference for the paper mentioned below: ^^^intriguingproperties^^^
argumentParser.add_argument("--szegedy-linear-modifier",
                            type=str,
                            help="A string that aids in the construction of \
                                  networks.SzegedyLinear; valid choices are \"FC-100-100-10\" and \
                                  \"FC-123-456-10\", both of which can be found in the paper at \
                                  https://storage.googleapis.com/pub-tools-public-publication-data/\
                                  pdf/42503.pdf",
                            default="FC-100-100-10")

# chainer.functions is here:^^^chainerfunctions^^^.
argumentParser.add_argument("--activation",
                            type=str,
                            help="Which function (relative to chainer.functions or \
                                 networkcomponents) that you would like to use as a non-linearity \
                                 (although two options in --layer are technically non-linearities \
                                 themselves, but we mean \"non-linearity\" in the canonical \
                                 sense). Not literally every function under chainer.functions will \
                                 work. \"relative\" means that the value that you put into here \
                                 works the same way as those used for --classification-loss-term \
                                 and --other-loss-term. If this option isn't used, then the \
                                 network will have no actiation function",
                                 default="none")

# Input strings for this option are the names of their classes:
# "Convolution2D"^^^chainerconvolution2d^^^, and "PairwiseDifference" and "Angular", which can be
# found in networkcomponenets.py)
argumentParser.add_argument("--layer",
                            type=str,
                            default="Convolution2D",
                            help="Which layer you want the network to use (up until the last \
                                 layer, which will be fully-connected). Can be either \
                                 chainer.links.Convolution2D, \
                                 networkcomponents.PairwiseDifference, or \
                                 networkcomponents.Angular; Their respective values for this \
                                 option should be listed with this help message",
                            choices=("Angular", "Convolution2D", "PairwiseDifference"))

#argumentParser.add_argument("algorithm",
#                            type=str,
#                            choices=["stochastic"],
#                            help="What you want to use to optimize the network")

# This argument allows the user to choose the dataset they want to use (MNIST^^^mnist^^^ or
# CIFAR-10^^^cifar^^^). See below during dataset setup for the reason for "mnistinverted".
argumentParser.add_argument("trainOn",
                            type=str,
                            help="Choose a dataset; \"mnistinverted\" simply means that pixels are \
                                  inverted such that, as you get closer to 0, the image gets \
                                  brighter (this is because it was the original way MNIST was to \
                                  be interpreted according to http://yann.lecun.com/exdb/mnist/)",
                            choices=["mnist", "mnistinverted", "cifar10"])

# Both of these options use the idea (probably from ^^^missing^^^) of passing in compound options
# where one or more additional settings for the specific option is included with the option itself
# in the same string.
####################################################################################################
#                                                                                                  #

def deCompound(s):
    match = re.match("(?P<weight>(?a:\d*\.\d*))(?P<function>\w+)", s)
    if match != None:
        option = match.groupdict()
        option["weight"] = float(option["weight"])
        return option
    else:
        raise Exception("Invalid option value used: {}".format(s))

# See ^^^chainermeansquarederror^^^; this is the full reference for
# https://docs.chainer.org/en/stable/reference/generated/chainer.functions.mean_squared_error.html.
# Further, chainer.functions can be found at ^^^chainerfunctions^^^.
argumentParser.add_argument("--classification-loss-term",
                            type=deCompound,
                            action="append",
                            help="The loss term you want to use that penalizes based on mispredicti\
                                 on (can put multiple of these flags on the command line). \
                                 Must be the name of a callable in the \"functions\" submodule of \
                                 Chainer or a callable from tackons.py that has the same \
                                 signature as chainer.functions.mean_squared_error(...). Provide a \
                                 floating-point weight just before the name to give a weight to \
                                 the respective loss emitted")

# Help message URL is this reference: ^^^chainerfunctions^^^.
argumentParser.add_argument("--other-loss-term",
                            type=deCompound,
                            action="append",
                            help="Any loss term desired that doesn't fall into the category \
                                 that would be entered with --classification-loss-term. Should be \
                                 the name of a callable from tackons.py or \
                                 https://docs.chainer.org/en/stable/reference/functions.html. \
                                 Otherwise everything else is the same as in \
                                 --classification-loss-term")

#                                                                                                  #
####################################################################################################


# As the help statement says, Batch Renormalization^^^batchrenormalization^^^ can be turned on with
# this option.
argumentParser.add_argument("--renormalize",
                            action="store_true",
                            help="The trainer should use Batch Renormalization \
                                 (https://arxiv.org/pdf/1702.03275.pdf)",
                            default=False)

# See ^^^chainergetdevice^^^ to find out about what needs to be passed to this argument (the "device
# spec").
argumentParser.add_argument("--device",
                            type=str,
                            default="@numpy",
                            help="Lets you select the device on which you want to perform the \
                                 training. Use a string specified in https://docs.chainer.org/en/st\
                                 able/reference/generated/chainer.get_device.html")

# "meta information" document is the same as settings.json from Counting
argumentParser.add_argument("trainingProducts",
                            type=str,
                            help="The save location of the file that contains meta information of \
                                 the training as well as anything that's saved by a Trainer from \
                                 https://docs.chainer.org/en/stable/reference/generated/chainer.tra\
                                 ining.Trainer.html")

argumentParser.add_argument("--learning-rate",
                            type=float,
                            help="The learning rate to use during optimization when applicable",
                            default=0.0004)

argumentParser.add_argument("--epochs",
                            type=int,
                            help="Used to specify how long the training runs (in epochs), but it   \
                                 may terminate sooner if the loss hasn't improved in 3 epochs",
                            default=1000)

argumentParser.add_argument("--notes",
                            type=str,
                            help="Just a place to write things you want to keep for this training \
                                 session")

argumentParser.add_argument("--no-improvement-for",
                            type=int,
                            help="Number of epochs over which there is no improvement of the model \
                                 before training is stopped",
                            default=5)

argumentParser.add_argument("--fraction-validation",
                            type=float,
                            default=0.07,
                            help="How much of the training dataset you'd like to devote to \
                                 validation. Actual sample count will be rounded as necessary.")

#   --perturbation-norm and --adversarial-training-with refer to the usage of adversarial training,
# which (I believe) is from ^^^explaining^^^, and uses either PGD from ^^^towards^^^ or
# FGSM^^^explaining^^^ as training adversaries.
####################################################################################################
#                                                                                                  #

# "https://arxiv.org/pdf/1706.06083.pdf" refers to ^^^towards^^^. Valid inputs were made to be the
# same as that of testbench.py.
argumentParser.add_argument("--adversarial-training-with",
                            type=str,
                            help="Whether or not adversarial training should be done. This uses \
                                 the adversarial training described in \
                                 https://arxiv.org/pdf/1706.06083.pdf of part of which requires \
                                 training against adversarial examples from either FGSM or PGD, \
                                 FGSM being from https://arxiv.org/pdf/1412.6572.pdf and PGD being \
                                 from https://arxiv.org/pdf/1706.06083.pdf; you specify which with \
                                 this option",
                            choices=["FGSM", "PGD"])
argumentParser.add_argument("--perturbation-norm",
                            type=int,
                            default=20,
                            help="The step size of the addition of the sign of the gradient in \
                                 FGSM or the maximum L-infinity noise allowed when using PGD")

parameters =  argumentParser.parse_args()

#                                                                                                  #
####################################################################################################

#                                                                                                  #
####################################################################################################



# Setting up each component picked by the user
####################################################################################################
#                                                                                                  #

lossFunction                   = common.ArbitraryLoss(
                                    parameters.classification_loss_term,
                                    parameters.other_loss_term
                                 )
net                            = common.create(
                                    vars(parameters),
                                    common.imageMetadata[parameters.trainOn]["channels"],
                                    common.imageMetadata[parameters.trainOn]["width"],
                                    common.imageMetadata[parameters.trainOn]["height"]
                                 )
includingLoss                  = networks.ArbitraryLossClassifier(net, lossFunction)

#                                                                                                  #
####################################################################################################


# These datasets are loaded in with integer pixels (to hardware precision) as we will be following
# the mindset outlined in ^^^explaining^^^ that adversarial noise must be integral as cameras and
# the like capture images using integer values. Therefore, I believe they imply that any real-world
# attack would be using the aforementioned kind of noise. We will be doing that within this project.
# This section used to check if the network was linear or convolutional, and tailored the dataset
# dimensionality to fit each of those paradigms. For the sake of black-box attacks (where there
# can be a mixture of one convolutional and one linear, or vice versa), since all the networks in
# networks.py are technically convolutional, we can just stick with four-dimensional data (which
# corresponds to the 3 used in this section).
####################################################################################################
#                                                                                                  #

dataPairs  = None

# Loading MNIST^^^mnist^^^ the erroneous way (clarified in README)
if parameters.trainOn == "mnistinverted":
    dataPairs = chainer.datasets.get_mnist(scale=255,
                                           ndim=3,
                                           withlabel=True)
# The user chose CIFAR-10^^^cifar^^^ to train on
elif parameters.trainOn == "cifar10":
    dataPairs = chainer.datasets.get_cifar10(withlabel=True,
                                             ndim=3,
                                             scale=255)

#                                                                                                  #
####################################################################################################

training = dataPairs[0]
datasets = chainer.datasets.split_dataset_random(
                training,
                round(parameters.fraction_validation * len(training))
           )
tDataset = datasets[1]
vDataset = datasets[0]

# PIL.Image.fromarray(tDataset[0][0]).save("numbers.png")
# PIL.Image.fromarray(tDataset[1][0]).save("numberss.png")
# PIL.Image.fromarray(tDataset[2][0]).save("somany.png")
# PIL.Image.fromarray(tDataset[3][0]).save("numbersss.png")

trainingSerialIterator   = chainer.iterators.SerialIterator(tDataset, 20)
validationSerialIterator = chainer.iterators.SerialIterator(vDataset, 200, repeat=False)

# Using Stochastic Gradient Descent with momentum for now
optimizer = chainer.optimizers.MomentumSGD(parameters.learning_rate, 0.85)
optimizer.setup(includingLoss)

standardUpdater = chainer.training.updaters.StandardUpdater(
                        trainingSerialIterator,
                        optimizer,
                        device=parameters.device,
                        converter=common.AdversarialTrainingConverter(
                                    net,
                                    lossFunction,
                                    parameters.perturbation_norm,
                                    getattr(attacks, parameters.adversarial_training_with)
                                  ) if parameters.adversarial_training_with != None \
                                            else common.oneHotConverter
                  )

trainUntil = chainer.training.triggers.EarlyStoppingTrigger(
                max_trigger=(parameters.epochs, "epoch"),
                verbose=True,
                patience=parameters.no_improvement_for,
                mode="min",
                monitor="Training Error"
             )

loop = chainer.training.Trainer(standardUpdater,
                                trainUntil,
                                parameters.trainingProducts)

loop.extend(chainer.training.extensions.ProgressBar())

# Printing the loss from the optimizer, an idea from Counting/learn.py
####################################################################################################
#                                                                                                  #

toPrint = ["Training Error", "Training Error (class)", "Performance"]
loop.extend(
    chainer.training.extensions.PrintReport(
        toPrint,
        chainer.training.extensions.LogReport(toPrint, (500, "iteration"))
    ),
    trigger=(1, "epoch")
)

#                                                                                                  #
####################################################################################################

loop.extend(chainer.training.extensions.snapshot(), trigger=(10, "epoch"))
# Passing in a Classifier that run the accuracy calculation in the spirit of Standard
# Updater^^^chainerstandardupdater^^^
loop.extend(
    chainer.training.extensions.Evaluator(
        validationSerialIterator,
        chainer.links.Classifier(net, accuracyWrapper),
        device=parameters.device
    )
)

loop.extend(RMaxDMaxModifier.RMaxDMaxModifier())

# This file is the JSON file described above the code that creates the "trainingProducts" argument
# for argumentParser, and we save the attributes of "parameters" into it, an idea from
# Counting/learn.py too.
handle = open(pathlib.Path(parameters.trainingProducts, "meta.json"), "w")
json.dump(vars(parameters), handle)
handle.close()

with chainer.using_device(parameters.device):
    loop.run()


# Don't need to name this file after the type of network or anything else as these details should be
# in meta.py (see above lines that save "meta.json")
chainer.serializers.save_npz(  pathlib.Path(parameters.trainingProducts, "finalnetwork.npz"), net  )





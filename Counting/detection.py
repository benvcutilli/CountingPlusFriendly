# This file performs localization via a modification of ^^^saliency^^^'s method called a "saliency
# map". That method performs backpropagation from the output of the network to the image, then, in
# the multi-channel-image case, for each pixel, finds the channel of that pixel whose absolute value
# is the max of all channels for that pixel, and displays that absolute value as an intensity on the
# map. In this file, backpropagation starts at the summation of the two outputs instead of the
# paper's method of using a single output.
#   After performing this step, we backpropogate from every value p' in this saliency map, M. This
# is done one at a time, in decreasing order of saliency, to the channels of each pixel p of the
# image. This creates a "second" derivative (called the "saliency derivative") in which each pixel's
# channel in the image is assigned a value that determines how much its value can change p's
# saliency. Then, taking a page again from saliency, we perform the same abosolute value + max
# calculation of saliency on these channels again, creating a saliency map M'. The hope is that any
# strong correlation (determined by the values of M' meeting a threshold; using a threshold was
# inspired by ^^^thresholding^^^) between a location in M and a location in M' means that they are
# locations where parts of the same shape is. This is because it could be the case that a change in
# value of pixel p means that the saliency of p' needs to be reduced or increased so that the shape
# count remains the same (or approximately the same) as the network would hopefully compensate for
# the modified pixel. Note that the thresholding occurs because it is likely that too many pixels
# have some correlation with the saliency value, meaning that all of them might be accidentally
# considered in a shape if we don't weed out the pixels that don't affect the saliency much.
# Performing this backpropogation from the single saliency value to all pixels *not already
# attributed to another value on the saliency map* should give us an idea of the subset of the
# remaining pixels that belong to a shape.
#   That whole step is repeated until we've exhausted all pixels that could be a candidate for a
# pixel in a shape (ignoring backpropogating from saliencies whose respective pixels have already
# been assigned to a group of pixels that could pertain to a shape in previous iterations of pixel
# grouping). Currently, the program puts out an image for each shape it finds, highlighting the
# pixel locations of p' and those that "belong" to p's shape.
#   There is a requirement for activation functions to be smooth and doubly-differentiable for this
# to work, which is further elaborated on in networks.py above the Convolutional class's defintion.

# chainer, PIL, pathlib, math, and argparse are packages from ^^^chainer^^^, ^^^pillow^^^,
# ^^^pythonpathlib^^^, ^^^pythonmath^^^, and ^^^pythonargparse^^^
####################################################################################################
#                                                                                                  #

import chainer, pathlib, math, argparse
from PIL import Image

#                                                                                                  #
####################################################################################################

import common



chainer.global_config.train = False


# Saliency calculation function (assumes var is already the derivative w.r.t the image)
def saliency(var):
    
    return chainer.functions.squeeze(
                chainer.functions.max(
                    chainer.functions.absolute(var),
                    1
                )
           )

# Outputs a saliency^^^saliency^^^ image. Also works for the saliency of the saliency.
def makeImage(saliency, where):
    # Normalize between 0 and 255 (rounding as these should be integers). This normalization is
    # probably similar to, if not exactly, what ^^^saliency^^^ does (they don't specify the
    # specifics). Higher saliency gets a higher number, as this is what they appear to do in
    # ^^^saliency^^^. They also output greyscale images, so we do that too.
    ################################################################################################
    #                                                                                              #
    
    denominator           = chainer.functions.max(saliency)
    nonImageNormalization = saliency / denominator
    output                = chainer.numpy.around(nonImageNormalization.data * 255)                 \
                                                                    .astype(chainer.numpy.uint8)

    #                                                                                              #
    ################################################################################################
    
    data = Image.fromarray(output)
    data.save(where)












argumentParser = argparse.ArgumentParser()
argumentParser.add_argument("networkLocation", type=str, help="Where the network is saved")
argumentParser.add_argument("settingsLocation",
                           type=str,
                           help="Path to the settings.json file created by learn.py")
argumentParser.add_argument("imageLocation",
                            type=str,
                            help="The image that you would like to perform localization on")
arguments = argumentParser.parse_args()









# Loading image and network
####################################################################################################
#                                                                                                  #

imageVariable = chainer.Variable(
                    chainer.numpy.expand_dims(
                        chainer.numpy.array(
                            Image.open(arguments.imageLocation),
                            dtype=chainer.numpy.float32
                        ).transpose(2,0,1),
                        0
                    )
                )
network       = common.recreateNetwork(arguments.networkLocation, arguments.settingsLocation)

#                                                                                                  #
####################################################################################################


# Performing the saliency calculation step and saving it as an image (both of which are done in
# ^^^saliency^^^)
####################################################################################################
#                                                                                                  #

output              = network(imageVariable)
                      # The "* <number>" is inspired by multiplying the loss by something before
                      # backpropagation
gradientWRTimage    = chainer.grad((chainer.functions.sum(output),),
                                   (imageVariable,),
                                   enable_double_backprop=True)[0]
original            = saliency(gradientWRTimage)
# See the comment for the makeImage call towards the bottom of this file
makeImage(original ** 0.1, "saliency.png")
print(original[436:440,408:412])

#                                                                                                  #
####################################################################################################

print(chainer.numpy.argsort(original.data, axis=None))

# This is a value that we use to disregard pixels that have low importance to a saliency pixel.
saliencySaliencyThreshold   = 5e-10







# Backpropogating from appropriate saliencies and finding relevant pixels
####################################################################################################
#                                                                                                  #


# Keeping track of which pixels have not been attributed to a shape
accountedFor = chainer.numpy.zeros(original.shape, dtype=chainer.numpy.bool_)

# Are we done looking at all pixels in the image?
fullyExamined = accountedFor.all()

counter       = 1

print(output)

maxShapesToDetect = round(chainer.functions.sum(output).item())
while (not fullyExamined) and (counter <= maxShapesToDetect):
    # This code pattern of calling unravel_index(...)^^^numpyunravel^^^ on the return value of
    # argmax(...)^^^numpyargmax^^^ is not my idea; it is probably from ^^^unravelargmax^^^
    # or somewhere on ^^^numpy^^^'s website, but the date of retrieval is unknown.
    ################################################################################################
    #                                                                                              #

    zero            = chainer.numpy.zeros(original.shape)
    # The first parameter here sets the saliency of previously-used maxima and those maxima's
    # relatives to 0; this removes them from contention for the maximum (well, unless all the
    # saliencies are zero)
    where           = chainer.numpy.argmax(
                            chainer.numpy.where(accountedFor, zero, original.data),
                            axis=None
                      )
    where2d         = chainer.numpy.unravel_index(where, original.data.shape)
    maxX            = where2d[1]
    maxY            = where2d[0]
    
    #                                                                                              #    
    ################################################################################################

    
    
    # Saliency of saliency
    ################################################################################################
    #                                                                                              #

    maximum         = original[maxY][maxX]        
    backpropogation = chainer.grad((maximum,), (imageVariable,))
    derivative      = backpropogation[0]
    importance      = saliency(derivative)

    #                                                                                              #
    ################################################################################################

    # Which pixels are connected to the pixel with the maximum value?
    ################################################################################################
    #                                                                                              #

    passThreshold   = importance.data >= saliencySaliencyThreshold
    connected       = chainer.numpy.logical_and(
                        passThreshold,
                        chainer.numpy.logical_not(accountedFor)
                      )

    #                                                                                              #
    ################################################################################################
    
    
    # Removing those pixels deemed important by the threshold from future consideration, as
    # well as the one at (x, y) as that one is the "founder" of the group, so to speak
    ################################################################################################
    #                                                                                              #
             
    accountedFor             |= connected
    accountedFor[maxY][maxX]  = True
            
    #                                                                                              #
    ################################################################################################

    # Saving the binary image marking pixels hopefully relevant to a single shape
    Image.fromarray(connected).save(str(counter) + ".png")
    
    # Debug (I forgot what inspired me to use power here to draw small and large values close; maybe
    # log-scale graphs?)
    makeImage(  importance ** 0.1,   str(counter) + "-saliency.png"  )
    
    counter += 1

    fullyExamined = accountedFor.all()
      
#                                                                                                  #
####################################################################################################



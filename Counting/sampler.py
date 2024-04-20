# This program uses Shapely[863f65] at the recommendation of [40ceda, radouxju's answer] in order to
# see if a shape intersects other shapes. It just so happens that it is also able to manipulate
# shape properties as well (which you can see in the code). This file creates images with triangles
# and squares, the latter of which might be an idea attributed to [3e46ce].

# A: Could do a sampling of a continuous random variable here, but discrete is probably good enough
# for rotational purposes. Probably the same logic applies for scale as well


# Python's "sys"[a9bde3] module
import sys

# Python's "argparse"[38e732] module
import argparse

# Python's "random"[f14ddd] module
import random

# Python's "collections"[658323] module
import collections

# "shapely" refers to [863f65]
import shapely.geometry, shapely.affinity

# Pillow[699ba0]'s PIL
import PIL.Image, PIL.ImageDraw

# "math"[7c779f] is imported here
import math

# Module from [889500]
import pathlib

# json[61fceb] module
import json

def makeSampleFiles(controls):
    # The parameters that were passed into the command. See comments at the end of this file for the
    # creation of variable "x" for the reasons/uses of these options.
    ################################################################################################
    #                                                                                              #

    saveTo                = controls.saveTo
    numSamples            = controls.numSamples
    shapeLimit            = controls.shapeLimit
    imageHeight           = controls.image_height
    imageWidth            = controls.image_width
    shapeTypes            = controls.shape_types
    noShapeRotations      = controls.no_shape_rotations
    noShapeScaling        = controls.no_shape_scaling
    scaleBounds           = controls.scale_bounds
    shapeColor            = tuple(controls.shape_color)
    limitScaleResampling  = controls.limit_scale_resampling

    #                                                                                              #
    ################################################################################################

    # This holds templates for each kind of shape this program generates. The templates are used as
    # the base properties of an individual instance of a shape, and this shape is scaled by the
    # scaling factor. This results in shapes of many sizes.
    templates        = {}
    templates["tri"] = shapely.geometry.Polygon([(0,0), (1,0), (0,1)])
    templates["sq"]  = shapely.geometry.Polygon([(0,0), (1,0), (1,1), (0,1)])

    random.seed()

    # Will be used to create the file that contains ground truths for all the images
    truthDictionary = {}


    # We generate every possible shape within the maximum number of shapes specified; this was
    # changed from randomly sampling the shape counts and making sure they add up to within
    # shapeLimit as that may have created non-uniform distributions (with the hope that this does
    # generate a uniform distribution of shape counts after the sampling step described next). After
    # that, we sample from this list uniformly to get the sample count desired. The benefits of
    # doing this method is that the models train on more representative data. Further, testing on
    # such a dataset would avoid the problem of biased shape counts in a testing dataset, which
    # would not rigorously testing the model.
    ################################################################################################
    #                                                                                              #

    counts = []
    if "tri" in shapeTypes and "sq" in shapeTypes:
        for tri in range(shapeLimit+1):
            for sq in range(shapeLimit+1 - tri):

                counts.append({"tri": tri, "sq": sq})
    else:
        singleShape = shapeTypes[0]
        for howMany in range(shapeLimit+1):
            counts.append({singleShape: howMany})

    randomCounts   = random.choices(counts, k=numSamples)

    #                                                                                              #
    ################################################################################################


    for imageNumber in range(numSamples):

        numShapes = randomCounts[imageNumber]

        # Holds the shapes before we paint them into the image
        shapes = shapely.geometry.MultiPolygon([])

        # Randomly will create shapes because otherwise a set order of picking shapes may (although
        # I am not 100% sure about this) influence the properties of shapes over the whole dataset
        # generated. The method of repeating a thing's representation for how many times it exists,
        # then uniformly drawing them from that list to come up with a random choice, may not be my
        # original idea; I think it most likely came from code I was looking at that was distributed
        # to students [ce8e49]; the class may have even been [e5d7be] when I was a
        # teaching assistant.
        ############################################################################################
        #                                                                                          #
        bag  = []
        if "tri" in numShapes:
            bag += ["tri" for i in range(numShapes["tri"])]
        if "sq"  in numShapes:
            bag += ["sq"  for i in range(numShapes["sq"])]
        random.shuffle(bag)

        #                                                                                          #
        ############################################################################################

        # See point A above for comments about which kind of distribution this exact line samples
        # with
        scale = random.randint(scaleBounds[0], scaleBounds[1]) if not noShapeScaling else \
                            (scaleBounds[0])


        # In this section, we don't try to squeeze in the shape into the image by resampling the
        # scale or orientation, as that may introduce sampling biases. We instead attempt a shape
        # with the same properties except for position, which is sampled over and over as many times
        # as it takes until it fits.
        ############################################################################################
        #                                                                                          #

        for randomShape in bag:

            # Point A also applies to this section. The assignment of "scale" is redundant on the
            # first iteration
            ########################################################################################
            #                                                                                      #

            rotation = random.randint(0, 359) if not noShapeRotations else 0
            if not limitScaleResampling:
                scale = random.randint(scaleBounds[0], scaleBounds[1]) if not noShapeScaling else \
                            (scaleBounds[0])

            #                                                                                      #
            ########################################################################################

            donePositioning = False
            while not donePositioning:
                width_offset   = random.randint(0, imageWidth)
                height_offset  = random.randint(0, imageHeight)

                rotated     = shapely.affinity.rotate(templates[randomShape], rotation)
                scaled      = shapely.affinity.scale(rotated, scale, scale)
                finalShape  = shapely.affinity.translate(scaled, width_offset, height_offset)

                contained = finalShape.bounds[0] >= 0 and \
                            finalShape.bounds[1] >= 0 and \
                            finalShape.bounds[2] < imageWidth and \
                            finalShape.bounds[3] < imageHeight

                if (not shapes.intersects(finalShape)) and contained:
                    donePositioning = True
                    shapes = shapely.geometry.MultiPolygon([*shapes.geoms, finalShape])

        #                                                                                          #
        ############################################################################################

        image         = PIL.Image.new("RGB", (imageWidth, imageHeight))
        drawingHandle = PIL.ImageDraw.Draw(image)

        for shape in shapes.geoms:
            drawingHandle.polygon(shape.exterior.coords, shapeColor, shapeColor)

        digits   = len(str(numSamples))
        fileName = "{0:0{1}}.png".format(imageNumber + 1, digits)
        image.save(pathlib.Path(saveTo, fileName))

        truthDictionary[fileName] = numShapes

    # Writing the file that contains the ground truth for all the sample files
    ################################################################################################
    #                                                                                              #

    truthFile = open(pathlib.Path(saveTo, "truths.json"), "w")
    json.dump(truthDictionary, truthFile, indent=3)
    truthFile.close()

    #                                                                                              #
    ################################################################################################




x = argparse.ArgumentParser()
# Using option syntax that I see commonly used from here below
x.add_argument("saveTo",         type=str, help="Directory that will contain the generated samples")
x.add_argument("numSamples",     type=int, help="How many samples you want to make")
x.add_argument("shapeLimit",     type=int, help="The maximum number of shapes in a sample")
# Using option syntax that I see commonly used from here below
x.add_argument("--image-height", type=int, help="Sample image height", default=1000)
x.add_argument("--image-width",  type=int, help="Sample image width",  default=1000)
x.add_argument("--shape-types",
               default="tri",
               choices=["tri", "sq"],
               nargs="+",
               help="Which shapes you want")
x.add_argument("--no-shape-rotations",
               action="store_true",
               default=False,
               help="Disable rotating shapes")
x.add_argument("--no-shape-scaling",
               action="store_true",
               default=False,
               help="Shapes will be the same size")
# This argument is used to address an issue where empty space pixels are counted (instead of
# intelligent shape detection) if the shapes are all the same size. This problem was pointed out by
# [a0256e].
x.add_argument("--scale-bounds",
               default=[10, 20],
               type=int,
               nargs="+",
               help="The (inclusive) range (the first number being the lower bound and the second \
                    number being the upper bound) in which a scale will be chosen for the purposes \
                    of scaling the shape. If you provide one number, it is assumed that you are \
                    using --no-shape-scaling, and the number you provide will be the multiple by \
                    which the shape will be resized. In both these scenarios, I believe the scale \
                    may not be exact because my guess is that the positions of shape vertices end \
                    up being integers, not floating point. If you don't provide this when opting \
                    for --no-shape-scaling, the first integer in the two-integer default value of \
                    this argument will be used")
# [0b1618] proposed this concept of selecting scale
x.add_argument("--limit-scale-resampling",
               action="store_true",
               help="Don't sample every time a new shape is created; use the same scale for the \
                    whole image. Idea from (https://arxiv.org/pdf/2101.01386.pdf).")
x.add_argument("--shape-color",
               default=(0, 0, 255),
               nargs=3,
               type=int,
               help="Takes the red, green, and blue (space separated in that order) values for RGB \
                    color representation used for the color of the shape")
makeSampleFiles(x.parse_args())
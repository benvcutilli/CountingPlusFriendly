# The Plotly^^^plotly^^^ package
import plotly

# Importing ^^^numpy^^^
import numpy




def sigmoid(x):
    return (1 + numpy.exp(-x)) ** -1









samplesPerDimension = 500
# Using numpy.linspace to create x and y values is from somewhere on ^^^plotly^^^'s website, most
# likely. It is a convenient way to do this, so that's why.
evaluationRange = numpy.linspace([-5, -5], [5, 5], samplesPerDimension, axis=1)


# Using the technique that I used from networkcomponents.py (PairwiseDifference) where one dimension
# is on the first axis and the other is on the second axis so that they can broadcast to create all
# permutations between the array of x values and the array of y values. Before broadcasting, we need
# to add a dimension to both the x vector and y vector, but at the beginning and end of them,
# respectively, which is also what happens in PairwiseDifference. However, this code doesn't
# actually broadcast, but it mimics broadcasting with the .repeat(...) calls.
####################################################################################################
#                                                                                                  #

x = numpy.expand_dims(evaluationRange[0], 0).repeat(samplesPerDimension, 0)
y = numpy.expand_dims(evaluationRange[1], 1).repeat(samplesPerDimension, 1)
evaluationPairs = numpy.stack([x, y], 2)

#                                                                                                  #
####################################################################################################

weights         = numpy.array([1, 1])
constant        = 1.0

# Calculating every combination for the three functions
dotProduct     = numpy.dot(evaluationPairs, weights)
cosine         =                                dotProduct                                         \
                                                    /                                              \
               ( numpy.linalg.norm(weights) * numpy.linalg.norm(evaluationPairs, axis=2) )
softenedCosine =                                dotProduct                                         \
                                                    /                                              \
               ( numpy.linalg.norm(weights) * numpy.linalg.norm(evaluationPairs, axis=2) + constant)



dotProductSurface     = plotly.graph_objects.Surface(
                            x=evaluationRange[0],
                            y=evaluationRange[1], z=sigmoid(dotProduct)
                        )

cosineSurface         = plotly.graph_objects.Surface(
                            x=evaluationRange[0],
                            y=evaluationRange[1], z=cosine
                        )

softenedCosineSurface = plotly.graph_objects.Surface(
                            x=evaluationRange[0],
                            y=evaluationRange[1], z=softenedCosine
                        )


figure  = plotly.graph_objects.Figure(
                softenedCosineSurface,
                layout={ "scene": { "aspectmode": "data" } }
          )

# "validate" left as True partially because I trust the default value listed in
# ^^^plotlyfigureshow^^^
figure.show(renderer="firefox")

#figure.write_image("graph.png", "png", 1200, 900, 1.0, True, "kaleido")
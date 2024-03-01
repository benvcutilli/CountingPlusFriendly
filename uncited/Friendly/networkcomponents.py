# This file holds the defensive layers. They can be selected to be put into a network.


# See ^^^chainer^^^ for this module's reference
import chainer

# The math^^^pythonmath^^^ module
import math

import common











# Idea defined by ^^^cutillielastic^^^
class ElasticSigmoid(chainer.Link):

    def __init__(self):

        super().__init__()

        with self.init_scope():
            # Training will eventually train this initial values, which would result in a normal
            # sigmoid, to something else
            self.scale = chainer.Parameter(1.0)
            self.scale.initialize(1)


    def forward(self, tensor):

        return chainer.functions.sigmoid(self.scale * tensor)




# New kinds of layers. † The activation function, that would be after the calculation of the
# potentials, is left out mostly to facilitate the modularity (in this case, of being able
# to choose an activation function) discussed above
####################################################################################################
#                                                                                                  #

# This class isn't currently used. Please ignore.
class PairwiseSingleElement(chainer.FunctionNode):

    def __init__(self):

        super().__init__()
        # Attempted CUDA^^^cuda^^^ code
        self.code =  'extern "C" {                                                                       \
                                                                                                         \
                         __device__ __forceinline__ float differenceSigmoid(first, second) {             \
                             return 1.0 / (1.0 + exp(first - second));                                   \
                         }                                                                               \
                                                                                                         \
                                                                                                         \
                         __device__ __forceinline__ calculatePosition(long long sample,                  \
                                                                      long long sampleStride,            \
                                                                      long long y,                       \
                                                                      long long yStride,                 \
                                                                      long long x) {                     \
                             // Either I\'ve seen this formula before (just to determine offset          \
                             // from stride) or I\'ve written it sometime before^^^missing^^^            \
                             return sample * sampleStride + y * yStride + x                              \
                         }                                                                               \
                                                                                                         \
                                                                                                         \
                         __global__ void rowWise(float* cuts,                                            \
                                                 long long cutSize,                                      \
                                                 float* weights,                                         \
                                                 float* out) {                                           \
                                                                                                         \
                             long long inputOffset  = calculatePosition(blockIdx.z,                      \
                                                                        blockDim.y * blockDim.x,         \
                                                                        blockIdx.y,                      \
                                                                        blockDim.x                       \
                                                                        blockIdx.x);                     \
                                                                                                         \
                             for (int i = 0; i < cutSize; i++) {                                         \
                                                                                                         \
                                 long long weightOffset = threadIdx.x * cutSize + i                      \
                                 float comparison = differenceSigmoid(                                   \
                                                        cuts[inputOffset + threadIdx.x],                 \
                                                        cuts[inputOffset + i]                            \
                                                    );                                                   \
                                 out[inputOffset] = fmaf(comparison,                                     \
                                                         weights[weightOffset],                          \
                                                         out[inputOffset]);                              \
                                                                                                         \
                             }                                                                           \
                                                                                                         \
                         }                                                                               \
                                                                                                         \
                                                                                                         \
                         __device__ __forceinline__ float sigmoidDerivative(float input) {               \
                             return input * exp(-input) / pow(1 + exp(-input), 2)                        \
                         }                                                                               \
                                                                                                         \
                         // "chain" is the gradient from layers that end up above whatever layer         \
                         // this function is called within                                               \
                         __global__ void (float* cuts,                                                   \
                                          long long cutSize,                                             \
                                          float* weights,                                                \
                                          float* chain,                                                  \
                                          float* outInputs,                                              \
                                          float* outNeurons,                                             \
                                          long long outNeuronsCount) {                                   \
                                                                                                         \
                            long long inputOffset  = calculatePosition(blockIdx.z,                       \
                                                                       blockDim.y * blockDim.x,          \
                                                                       blockIdx.y,                       \
                                                                       blockDim.x                        \
                                                                       blockIdx.x);                      \
                                                                                                         \
                            // This loop calculates the gradient for just the incoming activations       \
                            // See ^^^cutillipairwise^^^ to understand why the code does what it         \
                            // does                                                                      \
                            for (int n = 0; n < outNeuronsCount; n++) {                                  \
                                                                                                         \                                                             \
                                for (int i = 0; i < cutSize; i++) {                                      \
                                                                                                         \
                                    long long posWeightOffset = threadIdx.x * cutSize + i;               \
                                    long long negWeightOffset = i * cutSize + threadIdx.x;               \
                                    float wrtValue                  = cuts[threadIdx.x + inputOffset];   \
                                    float otherValue                = cuts[inputOffset + i];             \
                                                                                                         \
                                    float posComponent =   sigmoidDerivative(wrtValue, otherValue)       \
                                                                          *                              \
                                                                weights[posWeightOffset];                \
                                    float negComponent = - sigmoidDerivative(otherValue, wrtValue)       \
                                                                          *                              \
                                                                weights[negWeightOffset];                \
                                    outInputs[threadIdx.x + inputOffset] += posComponent + negComponent; \
                                                                                                         \
                                    outNeurons[posWeightOffset] += differenceSigmoid(wrt, otherValue)    \
                                    outNeurons[negWeightOffset] += differenceSigmoid(wrt, otherValue)    \
                                }                                                                        \
                            }                                                                            \
                         }                                                                               \
                                                                                                         \
                                                                                                         \
                      }'

    def forward_cpu(self, operands):
        cuts, weights = operands

    def forward_gpu(self, operands):
        cuts, weights = operands

    def backward(self, perRowGradient):
        perRowGradient = perRowGradient[0]



# A description of this layer can be found in ^^^cutillipairwise^^^
class PairwiseDifference(chainer.Link):

    # Accepts the parameters of chainer.functions.im2col(...)^^^chainerim2col^^^, but in keyword
    # argument form. These keyword arguments are held in toPass, and chainer.functions.im2col(...)
    # will receive them in forward(...).
    def __init__(self, channelsConsumed, channelsProduced, **toPass):

        super().__init__()

        with self.init_scope():
            xyCount      = None
            try:
                xyCount  = math.prod( toPass["ksize"] )
            except:
                xyCount  = toPass["ksize"] ** 2
            self.weights = chainer.Parameter(chainer.initializers.Uniform(1.0))
            self.weights.initialize(
               (    channelsProduced,      (xyCount * channelsConsumed) ** 2    )
            )

        self.toPass = toPass


    def forward(self, tensor):

        sections       = chainer.functions.im2col(tensor, **self.toPass)

        # Doing the same thing here that Angular does, which is putting the channels of an (x, y)
        # location into the last dimension (although in Angular the channels are in dimension 0; the
        # tensor dimensions are flipped) so it is easier to understand in my head
        rearranged     = sections.transpose((0, 2, 3, 1))

        subtraction    = chainer.functions.expand_dims(rearranged, 4)                              \
                                            -                                                      \
                         chainer.functions.expand_dims(rearranged, 3)

        subactivation  = chainer.functions.sigmoid(subtraction)

        # This call to reshape(...) is done to implement the manipulation technique (from
        # Angular.forward()) for the same reason stated in Angular.forward(...).
        multiplication = subactivation.reshape( (-1, self.weights.shape[1]) )   @   self.weights.T


        return multiplication.reshape( (*rearranged.shape[0:3], -1) ).transpose( (0, 3, 1, 2) )







# Implementation of ^^^cutilliangular^^^
class Angular(chainer.Link):

    # Same comment as for PairwiseDifference.__init__(...)
    def __init__(self, channelsConsumed, channelsProduced, **toPass):

        super().__init__()

        with self.init_scope():
            xyCount      = None
            try:
                xyCount  = math.prod( toPass["ksize"] )
            except:
                xyCount  = toPass["ksize"] ** 2
            self.weights = chainer.Parameter(chainer.initializers.Uniform(1.0))
            self.weights.initialize(
                (    channelsProduced,      xyCount * channelsConsumed    )
            )

            # Acts like a bias (and sorta meant to be one), but kinda isn't exactly the same if
            # Angular is not followed by an activation function
            self.lenience = chainer.Parameter(
                                chainer.numpy.zeros(
                                    (channelsProduced, 1),
                                    dtype=chainer.numpy.float32
                                )
                            )

        self.toPass = toPass


    def forward(self, tensor):

        # This part does matrix multiplication where the left-hand side are the weights and the
        # right-hand side are what we are multiplying the weights by (which is the usual ordering
        # convention). Decided to flatten "subactivation" before performing the matrix
        # multiplication because I think, according to ^^^chainermatmul^^^, it broadcasts the matrix
        # multiplication instead of doing one giant matrix multiplication. Thus, I suspect
        # broadcasting is not as fast, seeing as matrix multiplication is one way to reach peak
        # performance^^^p573^^^.
        ############################################################################################
        #                                                                                          #

        sections = chainer.functions.im2col(tensor, **self.toPass).transpose((1, 3, 2, 0))

        right          = chainer.functions.reshape(
                             sections,
                             (sections.shape[0], -1)
                         )
        left           = self.weights

        multiplication = left @ right

        #                                                                                          #
        ############################################################################################

        # The constant in the denominator has its rationale as a derivation in the README
        cosine =                          multiplication                                           \
                                                /                                                  \
        (       common.n_norm(2, left, (1,))  *  common.n_norm(2, right, (0,))   +   0.2     )     \
                                                +                                                  \
                                          self.lenience

        return cosine.reshape( (-1, *sections.shape[1:4]) ).transpose( (3, 0, 2, 1) )


#                                                                                                  #
####################################################################################################





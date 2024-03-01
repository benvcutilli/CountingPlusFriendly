# Package from ^^^chainer^^^
import chainer

# Python "math" module^^^pythonmath^^^
import math

# The collections module, found at ^^^pythoncollections^^^
import collections

import common


# A. This function follows the semantics of how Chainer wraps^^^wrappingandcalling^^^ (and maybe
#    ^^^missing^^^) its creation of instances of subclasses of FunctionNode^^^chainerfunctionnode^^^
#    and of how it then calls those instances' .apply(...) method with a Python function (and just
#    one Python function). The function is often named roughly the same as the FunctionNode subclass
#    it handles, so modifiedSoftPlus(...) is similar (except I use camel case instead of
#    underscores, therefore not all characters are lower case). One goal with this is to allow the
#    user to avoid the clumsiness of calling ModifiedSoftPlus.apply(...) with a tuple of
#    "Variable"s^^^chainervariable^^^ (which is probably one reason why Chainer does it with its
#    functions). The Chainer developers also probably found it useful to do it this way as the user
#    wouldn't have to deal with the constructor and apply(...), so I might have chosen to do it this
#    way because of that as well.
#       However, wrapping was desired (probably also a motivation for ^^^wrappingandcalling^^^/
#    ^^^missing^^^) mostly to avoid a bug where calling .apply(...) several times each with a
#    different variable in the tuple results in Chainer throwing an error in ^^^backproputils^^^;
#    this error occurs at the line
#       assert gx == []
#    in assert_no_grads(...) of GradTable. Prior to testing the code, I thought that doing this may have been an
#    issue because of the creation of the subclass of FunctionNode^^^chainerfunctionnode^^^ that
#    ^^^wrappingandcalling^^^ or ^^^missing^^^ does. Plus, using FunctionNode.apply(...) repeatedly
#    is said to not work by ^^^chainermultipleapplys^^^. We therefore duplicate the ModifiedSoftPlus
#    instance in this function, and call .apply(...) on that instance instead.
#       I had thought about making an instance of ModifiedSoftPlus callable a la PyTorch
#    layers^^^pytorchcallable^^^ and probably Chainer's Link^^^chainerlink^^^. However, my
#    implementation of that required the instance to copy itself to avoid calling apply(...)
#    multiple times with the same instance. This is an issue because ^^^pythoncopy^^^ notes that
#    deep copies will copy not just what you are interested in copying but also anything else
#    attached to that object. They state this increases memory consumption, which is true (and
#    something that I want to avoid), but I also am not sure what exactly happens when you make
#    copies of the graph Chainer creates.








# Softplus^^^softplus^^^, but using a different base during exponentiation and the logarithm. Made to
# prevent the overflow issue reported by Chainer that is described in the comment above
# Convolutional (by enabling trying a smaller base). "alternate" is the alternate base (as opposed
# to e in ReLU).
def pureChainerMSP(alternate, potentials):
    # Formula is normal change of base, whose existence I was reminded of by
    # ^^^changeofbasemethod^^^ (although I had forgotten what it was). Instead of using
    # Numpy^^^numpy^^^, however, we are using Chainer.
    return chainer.functions.log(1 + 1.1**potentials) / math.log(alternate, math.e)


# --------------------------------------------------------------------------------------------------


# Derivation of the derivatives calculated in .forward(...) and .backward(...) can be seen in the
# README.
class ModifiedSoftPlusDerivative(chainer.FunctionNode):

    # Comment above ModifiedSoftPlus.__init__(...) applies to this function as well
    def __init__(self, base):
        super().__init__()
        self.base = base
    
    def forward_cpu(self, potentialsTuple):
        raise Exception("This class cannot be used on a CPU")

    # See the comment in ModifiedSoftPlus.forward_gpu(...) for why the CUDA
    # kernel^^^kernelparadigm^^^ is defined in here
    def forward_gpu(self, potentialsTuple):
        # CUDA C++^^^gpulanguage^^^ code:
        code = "derivative = 1 / (1 + powf({}, -potential))"
        code = code.format(self.base)

        kernel = chainer.backends.cuda.elementwise("r potential",
                                                   "r derivative",
                                                   code,
                                                   "modified_softplus_derivative")
        self.retain_inputs((0,))
        result    = kernel(potentialsTuple[0])
        tupleized = (result, )
        return tupleized

    # Same logic with this function regarding the elementwise-derivative perspective as can be found
    # above ModifiedSoftPlus.backward(...). Chain rule (because of the elementwise-ness) means we
    # multiply the gradients backpropogated to this layer by the derivative calculated here.
    def backward(self, wrt, backpropogatedTuple):

        internalDerivative = math.log(self.base) * self.base**(-self.get_retained_inputs()[0])     \
                                                    /                                              \
                               ( 1 + self.base ** (-self.get_retained_inputs()[0]) ) ** 2
                             
        return ( internalDerivative * backpropogatedTuple[0], )


# --------------------------------------------------------------------------------------------------


# This does the same thing as pureChainerMSP(...), it just has code specifically for the GPU. The
# calculation of the activation function is the same as defined in pureChainerMSP(...).
class ModifiedSoftPlus(chainer.FunctionNode):

    # "base" is the base we are using in the Modified SoftPlus. We are going to take a cue from the
    # Convolutional* classes below and store our desired base as an attribute to use later.
    def __init__(self, base):
        super().__init__()
        self.base = base


    def forward_cpu(self, potentialsTuple):
        return (pureChainerMSP(self.base, potentialsTuple[0]),)
    
    def forward_gpu(self, potentialsTuple):
        # Putting the kernel definition in here is an idea from ^^^chainerkernels^^^. Although not a
        # solution proposed by that reference for this specific problem (at least that I can
        # remember), this ended up being convenient in case there is no CUDA^^^cuda^^^ compiler on
        # the system to compile the kernel, so putting it in __init__(...) wouldn't make sense as
        # all systems would run it. Also, this code is CUDA C++^^^gpulanguage^^^.
        code = "activation = logf(  1 + powf({0}, potential)  ) / logf({0})"
        kernel = chainer.backends.cuda.elementwise("r potential",
                                                   "r activation",
                                                   code.format(self.base),
                                                   "modified_softplus")
        self.retain_inputs((0,))
        return ( kernel(potentialsTuple[0]), )




    # I don't remember how, exactly, but one or more of the derivatives in
    # ^^^matrixcookbookderivatives^^^ showed me, not literally, that elements in matrices aren't any
    # more special than if those elements were taken out of a matrix and put in as parameters in an
    # ordinary equation. This means that you only need to figure out what value each element
    # influences, and in this case, the activation potential values of the tensor being passed in
    # influences just its respective element in the output tensor of the function.
    def backward(self, wrt, backpropogatedTuple):
        
        backpropogated = backpropogatedTuple[0]
    
        # Determining which device the items of "backpropogatedTuple" is on because we are
        # definitely gonna use those gradients to calculate the returned by this function gradient,
        # as discussed in this method's comment
        if isinstance(backpropogated.device, chainer.backend.GpuDevice):
            derivativeFunction = ModifiedSoftPlusDerivative(self.base)
            potentials         = self.get_retained_inputs()[0]
            partialDerivatives = derivativeFunction.apply((potentials, ))[0] * backpropogated
            return (partialDerivatives, )
        else:
            # Chainer should have already done the partial derivative work for us through the
            # use of its functions used in pureChainerMSP(...), so since self.backward(...) is called
            # in the order determined by the chain rule, the effect of this function is to return
            # the identity, which we do here
            return backpropogatedTuple

# --------------------------------------------------------------------------------------------------


# See (A) for why this function exists
def modifiedSoftPlus(base, exponent):
    node = ModifiedSoftPlus(base)
    t    = node.apply((exponent,))
    return t[0]
 
 
# --------------------------------------------------------------------------------------------------


# My version of Batch Normalization^^^batchnormalization^^^ instead of using the Chainer
# variant^^^chainerbatchnormalization^^^. This modifies the technique of Chainer's implementation of
# keeping running averages for the mean and standard deviation (actually, I think they store
# variance instead) to just calculating those averages for the last n batches trained on at
# serialization time. This class saves the initialization of self.expectation,
# self.standardDeviation, self.scale, and self.push for after __init__(...), which is what
# ^^^chainerbatchnormalization^^^ does, and I find that really convenient. As in
# chainer.links.BatchNormalization, self.gamma and self.beta are initialized in .forward(...), while
# I moved the initialization/calculation of self.expectation and self.standardDeviation to only
# occur in .serialize(...) in hopes of matching the efficiency of chainer.links.BatchNormalization's
# running averages; calculating these on every iteration would likely be very computationally
# intensive.
#   This layer assumes convolution dot product + bias is passed to it (see Convolutional for an
# explanation on why this is), and therefore averages over each neuron's potential map and the
# batch, as prescribed in the paper.
class BatchNormalizationCustom(chainer.Link):

    def __init__(self, n):

        super().__init__()

        with self.init_scope():
            # "Gamma" and "Beta" from ^^^batchnormalization^^^. Waiting until the first use
            # of this layer to generate the parameters in these variables is what
            # ^^^chainerbatchnormalizationcode^^^ does (or some other class in Chainer, possibly
            # ^^^chainerlinearcode^^^), so this may be why I do this here (code not copied from
            # anywhere; the pattern would have been the only thing taken from there).
            self.scale = chainer.Parameter(chainer.numpy.array(1, dtype=chainer.numpy.float32))
            self.push  = chainer.Parameter(chainer.numpy.array(1, dtype=chainer.numpy.float32))

        self.expectation          = None
        self.standardDeviation    = None
        self.n                    = n

        # Holds the previous self.n batches (self.n of them at most). This will be treated as a FIFO
        # queue.
        self.previousBatches      = collections.deque()
        
        self.register_persistent("expectation")
        self.register_persistent("standardDeviation")



    
    def forward(self, potentials):
        
        batchExpectation       = self.expectation if chainer.global_config.train == False          \
                                 else self.calculateExpectation(potentials)
        # Passing in the previously-computed expectation for the efficiency reasons stated above
        # BatchNormalizationCustom.calculateStandardDeviation(...) in the else's expression
        batchStandardDeviation = self.standardDeviation if chainer.global_config.train == False    \
                                 else self.calculateStandardDeviation(potentials,batchExpectation)

        # Lazy-initializing self.scale and self.push if needed like BatchNormalization
        # ^^^chainerbatchnormalizationcode^^^ and/or ^^^chainerlinearcode^^^; using an if statement
        # to check for initialization is what I believe Chainer's BatchNormalization.forward(...)
        # does; if not, it was some other Chainer class.
        ############################################################################################
        #                                                                                          #
        
        if self.scale.is_initialized == False:
            self.scale.initialize(batchExpectation.shape)

        if self.push.is_initialized == False:
            self.scale.initialize(batchExpectation.shape)
        
        #                                                                                          #
        ############################################################################################

        # Collecting the .data attribute instead of the actual Variable itself was done to try
        # to cut down on memory usage to be more in line with how much memory
        # ^^^chainerbatchnormalization^^^ uses, which it did.
        if len(self.previousBatches) == self.n and chainer.global_config.train == True:
            self.previousBatches.append(potentials.data)
            self.previousBatches.popleft()
        elif chainer.global_config.train == True:
            self.previousBatches.append(potentials.data)

        
        return ((potentials - batchExpectation) / batchStandardDeviation) * self.scale + self.push
    
    
    def serialize(self, passthrough):
        
        # The code inside this if statement generates the expectation and standard deviations used
        # in evaluation scenarios
        if not isinstance(passthrough, chainer.Deserializer):
        
            allSamples             = chainer.functions.concat(self.previousBatches, 0)

            self.expectation       = self.calculateExpectation(allSamples).data
            self.standardDeviation = self.calculateStandardDeviation(allSamples,
                                                                     self.expectation).data


    
        super().serialize(passthrough)



    # I require the expectation to be passed in instead of calculating it again because I'm trying
    # to reduce the calculation time to something similar to how long it takes
    # chainer.links.BatchNormalization
    @classmethod
    def calculateStandardDeviation(cls, stacked, expectation):
        numElements     = stacked.shape[0] * stacked.shape[2] * stacked.shape[3]
        zeroExpectation = stacked - expectation
        variance        = chainer.functions.sum(zeroExpectation ** 2,
                                                axis=(0, 2, 3),
                                                keepdims=True)    /    numElements

        return chainer.functions.sqrt(variance)

  
    @staticmethod
    def calculateExpectation(stacked):
        numElements = stacked.shape[0] * stacked.shape[2] * stacked.shape[3]

        return chainer.functions.sum(stacked, axis=(0, 2, 3), keepdims=True) / numElements











 
 


# Counts the number of shapes in images that were generated by sampler.py. This network used to use
# ReLU^^^relu^^^ activations in the forward(...) function. However, this appeared to have presented
# a problem, as it was impossible to calculate the second derivative (since the Chainer
# implementation of ReLU^^^chainerrelu^^^ likely has the gradient piecewise 0 and 1 the way I was
# using it -- I can't remember where I heard of this technique^^^missing^^^), then the second
# derivative of those functions is 0, which, after applying the chain rule during backpropogation,
# would likely result in a 0 second partial derivative. This is only due to the fact that the second
# derivative is zero simply because it is actually not defined for 0 as its input, so any change in
# the derivative (the second derivative) cannot be captured from when the first derivative is 0 to
# when it changes to 1. Thus, a smooth activation function whose second derivative can be properly
# calculated at the very least throughout the output values that we would expect it to emit was
# chosen instead. I initially tried SoftPlus^^^softplus^^^, but that was giving me exceptions
# relating to overflow, so I switched to the sigmoid function. However, that was having trouble
# converging, so I created the above modifiedSoftPlus(...) and ModifiedSoftPlus to mimic SoftPlus
# but hopefully avoid the overflow issue.
#   To be more specific about overflow with SoftPlus, some point in the chain of functions, when
# calculating the second derivative as is done in localization.py, there is an exponentiation (e^x)
# that overflows.
#   Easily swapping out activation functions is desired for the reasons in the README, so that is
# why this class allows users to specify which activation function they want.
#   Usage of batch normalization^^^batchnormalization^^^ was recommended by ^^^alivaramesh^^^,
# which was actually a recommendation for the adversarial defenses part of this thesis; I just
# thought I'd find it useful here as I am also having trouble training sigmoid and sine-based
# activation functions (although I don't use sine in the other part's networks). However, batch
# normalization wasn't working during inference, so I went with Batch
# Renormalization^^^batchrenormalization^^^. Using ^^^chainerbatchrenormalization^^^ defaults for
# initial_{gamma, beta, avg_var, avg_mean} because they probably know better than I do.
#   The reason why convolutional layers are used everywhere except for in the last layer is because
# it is meant to be similar to ^^^cowc^^^'s InceptionV3^^^inception^^^-based network (the last
# layer being linear was not inspired by them).
class Convolutional(chainer.Chain):
    
    # countableShapes refers to the how many kinds of shapes are in the image, renormalize to
    # Batch Renormalization^^^batchrenormalization^^^
    def __init__(self,
                 countableShapes,
                 activationFunction,
                 baseForExponentiation,
                 renormalize=False):
        super().__init__()
        self.baseForExponentiation  = baseForExponentiation
        self.activationFunction     = activationFunction
        self.needsExtraArgs         = self.activationFunction in [pureChainerMSP, modifiedSoftPlus]
        self.countableShapes        = countableShapes
        self.renormalize            = renormalize

        with self.init_scope():

            # Turning on bias with what we passed into "renormalize", a technique taken from the
            # network constructors in Friendly/networks.py
            self.a = chainer.links.Convolution2D( 3, 10, 7, 5, 0, renormalize)
            self.b = chainer.links.Convolution2D(10, 10, 7, 5, 0, renormalize)
            self.c = chainer.links.Convolution2D(10, 10, 3, 5, 0, renormalize)

            self.d = chainer.links.Linear(None, countableShapes)
            
            # Dummy function that gets called if the user doesn't want Batch Renormalization
            def identity(thing):
                return(thing)
            
            # Trying 1 and 0 values for rmax and dmax, respectively, because
            # ^^^batchrenormalization^^^ suggests starting out with those; however, I am not
            # currently increasing it over time like they recommend. Using the default of 0.99 for
            # "decay" (even though we explicitly type it out here) because I am having accuracy
            # issues which may be due to my original value for decay, 0.95 (I assume that the
            # default is 0.99 because that is what works well).
            self.a_normalize = chainer.links.BatchRenormalization(decay=0.99,
                                                                  size=(10),
                                                                  rmax=1.0,
                                                                  dmax=0.0)                        \
                                                            if self.renormalize else identity
            self.b_normalize = chainer.links.BatchRenormalization(decay=0.99,
                                                                  size=(10),
                                                                  rmax=1.0,
                                                                  dmax=0.0)                        \
                                                            if self.renormalize else identity
            self.c_normalize = chainer.links.BatchRenormalization(decay=0.99,
                                                                  size=(10),
                                                                  rmax=1.0,
                                                                  dmax=0.0)                        \
                                                            if self.renormalize else identity


    
    def forward(self, volume):

        parameters = [self.baseForExponentiation] if self.needsExtraArgs else []
        
        # Don't quite understand why they recommend it, but ^^^batchnormalizationplacement^^^ says
        # it's better to normalize after the dot product; however, they omit the layer bias, but I
        # don't currently do that here (they point out the uselessness of bias when using batch
        # normalization in this order^^^batchnormalizationplacement^^^).
        ############################################################################################
        #                                                                                          #

        volume     = self.a_normalize(self.a(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        volume     = self.b_normalize(self.b(volume))        
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        volume     = self.c_normalize(self.c(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )
        
        #                                                                                          #
        ############################################################################################

        # Last layer _has_ to emit a number within the bounds [0, inf) because the output is a
        # count, so that is why ReLU^^^relu^^^ is chosen here
        count = chainer.functions.relu( self.d(volume) )

        return count


# Same as the above Convolutional network, except it has more layers and neurons, the rationale
# for which is in the comments of __init__(...)
class ConvolutionalLarge(chainer.Chain):
    
    # See comments for Convolutional.__init__(...)
    def __init__(self,
                 countableShapes,
                 activationFunction,
                 baseForExponentiation,
                 renormalize=False):
        super().__init__()
        self.baseForExponentiation     = baseForExponentiation
        self.activationFunction        = activationFunction
        self.needsExtraArgs            = self.activationFunction in [ pureChainerMSP,
                                                                      modifiedSoftPlus ]
        self.countableShapes           = countableShapes
        self.renormalize               = renormalize
        with self.init_scope():
            # The comments in __init__(...) of the Convolutional class above are relevant here. The
            # number of layers used was chosen because David Crandall^^^networksizecrandall^^^
            # suspected that the original network had too few layers to learn how to count two kinds
            # of shapes (he didn't suggest any specific number of layers to add). However, because
            # of the aforementioned change to the sigmoid activation function used in
            # Convolutional.forward(...) (which appears to have made the network incapable of
            # learning how to count triangles and squares at the same time) two things occurred to
            # follow Dr. Crandall's suggestion even further:
            #   1. the layers in the subsection "Sigmoid Compensation Layers" were added in addition
            #      to the other layers, and
            #   2. each layer was given successively more neurons than the last, until about the
            #      middle, where the neuron count decreases steadily until the end of the network
            #      (well, except for the output count, which is equal to the number of shape types we
            #      need to count); this build-up then build-down network structure is a method
            #      people use a lot, so I thought it would be useful here.
            # The additional layers were added to the middle-sh of the network as I believe it is
            # common to do so to give the network more power to learn intermediate representations,
            # which I think lack of such power could be the issue here. This modification didn't
            # seem to help, but I'm keeping it here because I imagine it can only help accuracy if
            # I can get the network to converge some other way. 
            ########################################################################################
            #                                                                                      #
            
            self.a = chainer.links.Convolution2D( 3, 30, 7, 3)
            self.b = chainer.links.Convolution2D(30, 40, 7, 3)
            self.c = chainer.links.Convolution2D(40, 50, 7, 3)
            
            # Sigmoid Compensation Layers
            ########################################################################################
            #                                                                                      #

            self.d = chainer.links.Convolution2D(50, 60, 5, 1, 2)
            self.e = chainer.links.Convolution2D(60, 70, 5, 1, 2)
            self.f = chainer.links.Convolution2D(70, 60, 5, 1, 2)

            #                                                                                      #
            ########################################################################################
            
            self.g = chainer.links.Convolution2D(60, 50, 5, 2)
            self.h = chainer.links.Convolution2D(50, 40, 5, 2)
            self.i = chainer.links.Convolution2D(40, 30, 3, 1)
            self.j = chainer.links.Convolution2D(30, 20, 3, 1)

            self.k = chainer.links.Linear(80, self.countableShapes)
            
            # Same comment for this as in Convolutional.__init__(...)
            def identity(thing):
                return(thing)
            
            # Same rationale for rmax and dmax values in these as in Convolutional.__init__(...)
            self.a_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(30),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.b_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(40),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.c_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(50),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.d_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(60),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.e_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(70),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.f_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(60),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.g_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(50),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.h_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(40),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.i_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(30),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            self.j_normalize = chainer.links.BatchRenormalization(decay=0.95, size=(20),
                                                                  rmax=1.0, dmax=0.0)              \
                                                            if self.renormalize else identity
            
            #                                                                                      #
            ########################################################################################

    
    def forward(self, volume):

        parameters = [self.baseForExponentiation] if self.needsExtraArgs else []
        
        # See respective comment in forward(...) of Convolutional
        ############################################################################################
        #                                                                                          #
  
        volume     = self.a_normalize(self.a(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        volume     = self.b_normalize(self.b(volume))        
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        volume     = self.c_normalize(self.c(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        volume     = self.d_normalize(self.d(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        volume     = self.e_normalize(self.e(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        volume     = self.f_normalize(self.f(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        volume     = self.g_normalize(self.g(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )
        
        volume     = self.h_normalize(self.h(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )
        
        volume     = self.i_normalize(self.i(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )
        
        volume     = self.j_normalize(self.j(volume))
        volume     = self.activationFunction(   *( parameters + [volume] )   )

        #                                                                                          #
        ############################################################################################



        volume     = chainer.functions.reshape(volume, (-1, 80))

        # See Convolutional.forward(...) for why we use ReLU here
        count = chainer.functions.relu( self.k(volume) )

        return count
# "chainer"[170ecf] package
import chainer

import networkcomponents

# Note 1: Every missing parameter of the BatchRenormalization[7ca151]
# constructor except dmax, use_gamma, rmax, and use_beta are assumed to be assigned to defaults that
# are at least somewhat optimal (I don't know what dtype does)
#
# Note 2: Following the strategy of [867913, section 3.2] to put batch renormalization
# between the "linear" part of a layer and the activation.


# [c214f9] has this network ("FC-100-100-10" and "FC-123-456-10"). † It is a
# universal class in that this class makes both networks (see parameter for __init__(...)) and
# allows the user to change which network they want to benchmark. It is here so that that paper's
# results can be compared to ours. However, we added in optional Batch
# Renormalization[e1d64c] layers (regular Batch
# Normalization[867913] was recommended by[16ccf1] to
# converge faster; I modified this suggestion by using Batch Renormalization instead). We are also
# using the technique from Yann LeCun[71bd98] where he said that you can treat a linear layers
# as convolutional ones that have a kernel size equal to the size of the input. † This allows us to
# use the non-activation, convolutional-style layers in networkcomponents.py, putting their classes
# in the layerType parameter of __init__ and calling them without needing to put conditionals into
# __init__ to handle each case of layer (however, we do need to specify our parameters on a
# layer-type-by-layer-type basis). † This also means that we need to change the linear layers out
# for convolutional ones that are basically linear due to the kernel size in order for this idea to
# work.
#   MalleableConvolutional has details on which dataset(s) it handles (SzegedyLinear works with the
# same datasets).
class SzegedyLinear(chainer.Chain):

    # "which" can be either "FC-100-100-10" or "FC-123-456-10". The comments for activationType,
    # layerType, and usingBatchRenorm in MalleableConvolutional.__init___(...) apply here; †
    # further, usingBatchRenorm also aids in contructing different kinds of networks. We need
    # "length1D" (the length in one of the two image dimensions) because of the switches outlined in
    # the black box-related comments above dataset loading in workbench.py
    def __init__(self,
                 depth,
                 length1D,
                 layerType=chainer.links.Convolution2D,
                 activationType=chainer.functions.sigmoid,
                 usingBatchRenorm=False,
                 which="FC-100-100-10"):

        super().__init__()

        isCustomLayer  = layerType in [
                                           networkcomponents.PairwiseDifference,
                                           networkcomponents.Angular
                                      ]
        # Different layers have different parameter names
        kwargNames  = {
            "in": "channelsConsumed" if isCustomLayer else "in_channels",
            "out": "channelsProduced" if isCustomLayer else "out_channels",
        }
        hyperparameters = self.getHyperparameters(depth,
                                                  length1D,
                                                  layerType,
                                                  which,
                                                  kwargNames,
                                                  isCustomLayer)


        construct = activationType == networkcomponents.ElasticSigmoid
        with self.init_scope():

            self.in0out1_preactiv = layerType(**hyperparameters["in0out1"])
            self.in0out1_activ    = activationType() if construct else activationType
            # Note 1 applies
            self.in0out1_renorm   = chainer.links.BatchRenormalization(
                                      (hyperparameters["in0out1"][kwargNames["out"]], 1, 1),
                                      decay=0.8
                                    ) if usingBatchRenorm else None


            layerType       = chainer.links.Convolution2D                                           \
                                  if layerType == networkcomponents.PairwiseDifference              \
                              else layerType
            isCustomLayer  = layerType in [
                                            networkcomponents.PairwiseDifference,
                                            networkcomponents.Angular
                                          ]
            # Re-evaluating kwargNames for the rest of the network
            kwargNames  = {
                "in": "channelsConsumed" if isCustomLayer else "in_channels",
                "out": "channelsProduced" if isCustomLayer else "out_channels",
            }
            hyperparameters = self.getHyperparameters(depth,
                                                      length1D,
                                                      layerType,
                                                      which,
                                                      kwargNames,
                                                      isCustomLayer)


            self.in1out2_preactiv = layerType(**hyperparameters["in1out2"])
            self.in1out2_activ    = activationType() if construct else activationType
            # Note 1 applies
            self.in1out2_renorm   = chainer.links.BatchRenormalization(
                                      (hyperparameters["in1out2"][kwargNames["out"]], 1, 1),
                                      decay=0.8
                                    ) if usingBatchRenorm else None

            self.final_preactiv   = chainer.links.Linear(
                                        hyperparameters["final"][kwargNames["in"]],
                                        hyperparameters["final"][kwargNames["out"]]
                                    )
            # They may have meant to have a SoftMax activation on the last layer, but it isn't clear
            # on page 8 of [c214f9], so I just go with the activation specified by
            # the user
            self.final_activ      = activationType() if construct else activationType
            # Note 1 applies
            self.final_renorm     = chainer.links.BatchRenormalization(
                                      (hyperparameters["final"][kwargNames["out"]],),
                                      decay=0.8
                                    ) if usingBatchRenorm else None





    # Parameters have same order and meaning as they do in __init__(...) except for kwargNames
    # and isCustomLayer, which are found in the body of __init__(...); name of kwargNames and
    # isCustomLayer is copied from __init__(...). † Called as part of constructing the network.
    def getHyperparameters(self, depth, length1D, layerType, which, kwargNames, isCustomLayer):

        # Simply getting the channel counts from "which". Just want to comment here that depth3
        # should always be 10, and you can find out why in a comment towards the bottom of
        # MalleableConvolutional.__init__(...)
        ############################################################################################
        #                                                                                          #

        depth1, depth2, depth3 = which.split("-")[1:]
        depth1, depth2, depth3 = (int(depth1), int(depth2), int(depth3))

        #                                                                                          #
        ############################################################################################

        hyperparameters  = {}

        hyperparameters["in0out1"] =  {
                                        # This entry was switched to a 2D kernel because, to
                                        # implement black box attacks datasets,
                                        # were switched to 4D
                                        "ksize": (length1D, length1D),
                                        kwargNames["in"]: depth,
                                        kwargNames["out"]: depth1,
                                      }
        # The "linear" outputs of the previous layer will all be in depth, so we only need a 1 x 1
        # kernel here
        hyperparameters["in1out2"] =  {
                                        "ksize": (1, 1),
                                        kwargNames["in"]: depth1,
                                        kwargNames["out"]: depth2,
                                      }
        # Same comment as above
        hyperparameters["final"]   =  {
                                        "ksize": (1, 1),
                                        kwargNames["in"]: depth2,
                                        kwargNames["out"]: depth3,
                                      }


        return hyperparameters




    def forward(self, data):

        # Originally added two dimension to "data" in this function before processing it with the
        # rest of this method, but then I removed it because I changed how the dataset is loaded so
        # that I can conduct black-box attacks.

        # Note 2 has relevant information for this section
        ############################################################################################
        #                                                                                          #

        temp  = self.in0out1_preactiv(data)
        if self.in0out1_renorm:
            temp = self.in0out1_renorm(temp)
        one   = self.in0out1_activ(temp) if self.in0out1_activ != None else temp

        temp  = self.in1out2_preactiv(one)
        if self.in1out2_renorm:
            temp = self.in1out2_renorm(temp)
        two   = self.in1out2_activ(temp) if self.in1out2_activ != None else temp

        temp  = self.final_preactiv(two)
        if self.final_renorm:
            temp = self.final_renorm(temp)
        three = self.final_activ(temp) if self.final_activ != None else temp

        #                                                                                          #
        ############################################################################################


        return three


















# Works for MNIST[e1b8ca] and CIFAR-10[c7fedb]; in theory, other datasets with 10 classes
# could work, but the second-to-last layer's output would be very large. See SzegedyLinear's comment
# which gives the rationale for Batch Renormalization[e1d64c].
class MalleableConvolutional(chainer.Chain):

    # † The "layerType", "activationType", and "usingBatchRenorm" parameters allows the user to
    # choose which type of neurons they want to use and their activation functions.
    def __init__(self,
                 depth,
                 layerType=chainer.links.Convolution2D,
                 activationType=None,
                 usingBatchRenorm=False):

        super().__init__()

        isCustomLayer      = layerType in [
                                            networkcomponents.PairwiseDifference,
                                            networkcomponents.Angular
                                          ]

        kwargNames  = {
            "in": "channelsConsumed" if isCustomLayer else "in_channels",
            "out": "channelsProduced" if isCustomLayer else "out_channels",
        }

        hyperparameters  = {}

        hyperparameters["in0out1"] = {
                                        "stride": 2,
                                        "ksize": 5,
                                        kwargNames["in"]: depth,
                                        kwargNames["out"]: 60,
                                     }
        hyperparameters["in1out2"] = {
                                        "stride": 1,
                                        "ksize": 3,
                                        kwargNames["in"]: 60,
                                        kwargNames["out"]: 50,
                                    }
        hyperparameters["in2out3"] = {
                                        "stride": 1,
                                        "ksize": 3,
                                        kwargNames["in"]: 50,
                                        kwargNames["out"]: 40,
                                     }
        hyperparameters["in3out4"] = {
                                        "stride": 1,
                                        "ksize": 3,
                                        kwargNames["in"]: 40,
                                        kwargNames["out"]: 30,
                                     }
        hyperparameters["in4out5"] = {
                                        "stride": 1,
                                        "ksize": 3,
                                        kwargNames["in"]: 30,
                                        kwargNames["out"]: 20,
                                     }
        hyperparameters["in5out6"] = {
                                        "stride": 1,
                                        "ksize": 3,
                                        kwargNames["in"]: 20,
                                        kwargNames["out"]: 10,
                                     }

        construct = activationType == networkcomponents.ElasticSigmoid
        with self.init_scope():
            self.in0out1_preactiv = layerType(**hyperparameters["in0out1"])
            self.in0out1_activ    = activationType() if construct else activationType
            # Note 1 applies
            self.in0out1_renorm   = chainer.links.BatchRenormalization(
                                      (hyperparameters["in0out1"][kwargNames["out"]], 1, 1),
                                      decay=0.8
                                    ) if usingBatchRenorm else None

            self.in1out2_preactiv = layerType(**hyperparameters["in1out2"])
            self.in1out2_activ    = activationType() if construct else activationType
            # Note 1 applies
            self.in1out2_renorm   = chainer.links.BatchRenormalization(
                                      (hyperparameters["in1out2"][kwargNames["out"]], 1, 1),
                                      decay=0.8
                                    ) if usingBatchRenorm else None

            self.in2out3_preactiv = layerType(**hyperparameters["in2out3"])
            self.in2out3_activ    = activationType() if construct else activationType
            # Note 1 applies
            self.in2out3_renorm   = chainer.links.BatchRenormalization(
                                      (hyperparameters["in2out3"][kwargNames["out"]], 1, 1),
                                      decay=0.8
                                    ) if usingBatchRenorm else None

            self.in3out4_preactiv = layerType(**hyperparameters["in3out4"])
            self.in3out4_activ    = activationType() if construct else activationType
            # Note 1 applies
            self.in3out4_renorm   = chainer.links.BatchRenormalization(
                                      (hyperparameters["in3out4"][kwargNames["out"]], 1, 1),
                                      decay=0.8
                                    ) if usingBatchRenorm else None

            self.in4out5_preactiv = layerType(**hyperparameters["in4out5"])
            self.in4out5_activ    = activationType() if construct else activationType
            # Note 1 applies
            self.in4out5_renorm   = chainer.links.BatchRenormalization(
                                      (hyperparameters["in4out5"][kwargNames["out"]], 1, 1),
                                      decay=0.8
                                    ) if usingBatchRenorm else None

            self.in5out6_preactiv = layerType(**hyperparameters["in5out6"])
            self.in5out6_activ    = activationType() if construct else activationType
            # Note 1 applies
            self.in5out6_renorm   = chainer.links.BatchRenormalization(
                                      (hyperparameters["in5out6"][kwargNames["out"]], 1, 1),
                                      decay=0.8
                                    ) if usingBatchRenorm else None


            # 10 classes for MNIST[e1b8ca] and CIFAR-10[c7fedb]
            self.final_preactiv = chainer.links.Linear(None, 10)
            self.final_activ    = activationType() if construct else activationType
            # See note 1, and also the 10 here has the same meaning as the "10" in the
            # above assignment of self.final_preactiv
            self.final_renorm     = chainer.links.BatchRenormalization(
                                      (10,),
                                      decay=0.8
                                    ) if usingBatchRenorm else None


    def forward(self, data):

        # Order of layers explained in note 2
        ############################################################################################
        #                                                                                          #

        temp  = self.in0out1_preactiv(data)
        if self.in0out1_renorm:
            temp = self.in0out1_renorm(temp)
        one   = self.in0out1_activ(temp) if self.in0out1_activ != None else temp

        temp  = self.in1out2_preactiv(one)
        if self.in1out2_renorm:
            temp = self.in1out2_renorm(temp)
        two   = self.in1out2_activ(temp) if self.in1out2_activ != None else temp

        temp  = self.in2out3_preactiv(two)
        if self.in2out3_renorm:
            temp = self.in2out3_renorm(temp)
        three = self.in2out3_activ(temp) if self.in2out3_activ != None else temp

        temp  = self.in3out4_preactiv(three)
        if self.in3out4_renorm:
            temp = self.in3out4_renorm(temp)
        four  = self.in3out4_activ(temp) if self.in3out4_activ != None else temp

        temp  = self.in4out5_preactiv(four)
        if self.in4out5_renorm:
            temp = self.in4out5_renorm(temp)
        five  = self.in4out5_activ(temp) if self.in4out5_activ != None else temp

        temp  = self.in5out6_preactiv(five)
        if self.in5out6_renorm:
            temp = self.in5out6_renorm(temp)
        six   = self.in5out6_activ(temp) if self.in5out6_activ != None else temp

        temp  = self.final_preactiv(six)
        if self.final_renorm:
            temp = self.final_renorm(temp)
        seven = self.final_activ(temp) if self.final_activ != None else temp

        #                                                                                          #
        ############################################################################################

        return seven












# Inspired by [e10c70] (roles of parameters of methods, the methods themselves,
# self.getLoss and self.network being the same as Classifer.loss, Classifier.predictor, maybe its
# superclass) except it calls an instance of common.ArbitraryLoss instead of the kinds of functions
# required by Classifier. Just like Classifier, however, you need to pass in to the constructor the
# thing that is doing the classification. † This is one of the only ways I am able to allow the user
# to easily add new components for the whole loss function to facilitate making Friendly a place to
# test different defenses. Possibly inspired by the functions and/or classes in attacks.py, the
# network is given to the ArbitraryLoss instance; the batch and ground truth is as well, but I can't
# remember if the idea is from the routines/classes in attacks.py as well.
class ArbitraryLossClassifier(chainer.Chain):

    def __init__(self, network, getLoss):
        super().__init__()
        self.getLoss = getLoss
        # I suspect that chainer.links.Classifier[e10c70] sets the passed-in network
        # as part of its learnable self. There appears to be a bug that occurs when that isn't done,
        # and it was fixed adding in code that does that fixes this
        with self.init_scope():
            self.network = network

    def forward(self, pixels, labels):
        # Constructor call in second parameter needed for functions in tackons.py that calculate
        # the gradient with respect to the input
        return self.getLoss(labels, chainer.Variable(pixels), self.network)





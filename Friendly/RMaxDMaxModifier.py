# This package is from [170ecf]
import chainer

# "collection" documentation at [658323]
import collections


# The code and comments (except for value "extensionclass" within any triple-caret-on-both-sides
# phrase and comments marked with a "New:" and any code that is marked as changed by a comment) in
# this section is from Counting/learn.py.
####################################################################################################
#                                                                                                  #

# Changes the values of rmax and dmax, which are attributes of
# chainer.links.BatchRenormalization[7ca151]. The rmax and dmax values
# of that class are set to increasingly higher values (though only slightly higher), as
# recommended by [e1d64c].
class RMaxDMaxModifier(chainer.training.Extension):

    def __init__(self):
        super().__init__()
    
        self.trigger = (1, "epoch")
        self.rmaxQueue = collections.deque()
        self.dmaxQueue = collections.deque()
    
        self.rmaxQueue.append(1.1)
        self.rmaxQueue.append(10)
        self.rmaxQueue.append(100)
        self.rmaxQueue.append(1000)
    
        self.dmaxQueue.append(1.1)
        self.dmaxQueue.append(10)
        self.dmaxQueue.append(100)
        self.dmaxQueue.append(1000)

        # New: Changed the value assigned here to an appropriate value for this attribute; the
        # original number works only in the context of Counting
        self.lossThreshold       = 0.01
        self.lossAchieved        = False
        self.lastEpoch           = 0
        self.epochsBetweenChange = 20

    # Using the same parameter name as chainer.training.Extension's[966bc2]
    # .__call__(...) just because I can't think of anything better
    def __call__(self, trainer):
    
        # New: Used the paradigm of getting the required information from the trainer and, in this
        # line, removed everything after the "trainer." and before the comparison operator,
        # replacing it with what you see here. I also maintained the behavior where just the last
        # iteration's loss is judged.
        case1 = ((trainer.observation["Training Error"].item() < self.lossThreshold) \
                                              and                                              \
                                    (not self.lossAchieved))

        case2 = (trainer.updater.epoch - self.lastEpoch >= self.epochsBetweenChange)           \
                                              and                                              \
                                        self.lossAchieved
                    
    
        if (case1 or case2) and len(self.rmaxQueue) != 0:
            dmaxReplacement = self.dmaxQueue.popleft()
            rmaxReplacement = self.rmaxQueue.popleft()
            # New: "network" in this line was a replacement for predictor in the original segment of
            # code in order for this class to work in this file
            for layer in trainer.updater.get_optimizer("main").target.network.children():
                if isinstance(layer, chainer.links.BatchRenormalization):
                    layer.rmax = rmaxReplacement
                    layer.dmax = dmaxReplacement
                    print("chainer.links.BatchRenormalization's dmax/rmax have been changed \
                          to {} and {}, respectively".format(dmaxReplacement, rmaxReplacement))

        if case1:
            self.lossAchieved = True
        if case1 or case2:    
            self.lastEpoch    = trainer.updater.epoch

#                                                                                                  #
####################################################################################################
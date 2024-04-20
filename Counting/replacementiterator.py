# module from [f14ddd]
import random
random.seed()

# The [170ecf] package
import chainer

# Package from [9f139d]
import multiprocessing

# [7c779f] module
import math





# This class was created because
# chainer.iterators.MultiprocessIterator[3e1f51] appears to have an issue
# with loading pieces of the dataset quickly enough.  I could be wrong, but I think
# MultiprocessIterator[2ea3fa] synchronizes all processes that are
# working on a single batch, which bottlenecks on the parent thread when processing because the
# fetched object for each thread becomes ready around the same time. However, you'd think that the
# worker processes would get out-of-sync enough for this to not be a problem after a little while. I
# dunno.
#   In this class, the workers aren't synchronized (mostly) with the parent. Any similarities
# between the two classes (besides those already stated) are pointed out throughout the class
# definition. This does the same thing as maybe [3e1f51]/PyTorch's
# DataLoader[448261, torch.utils.data.DataLoader] where a list of indices (self.indices) are shuffled instead of
# the dataset elements themselves so that not all elements of the dataset needs to be loaded into
# memory when the shuffle occurs.
#   As is likely case with [3e1f51] (and also its rationale), it doesn't
# matter where in the batch something is placed (as the loss is typically additive between the
# inputs, leading to the order-invariant addition of gradients using the chain rule), so no ordering
# from the shuffle is preserved in this class when assembling the batch. Following
# [ba7b10] and just general common practice, the last batch in an epoch will be short
# of the full batch size if not enough of the dataset is left.
class FreeFlowIterator(chainer.dataset.iterator.Iterator):
    
    # Putting the randomization of the dataset in this class is from the classes at
    # [3e1f51][f832fc]. "workers" takes the same name and role as its
    # respective parameter in the initialization function of those two classes as well (the name is
    # the same for consistency of use between the those classes, and this class). desyncLimit is how
    # many batches difference there is between the batch that __next___ is trying to get and the
    # latest batch that has been started in self.overflow before the training process gets killed.
    # This is done to prevent a stalled worker from stopping __next__ from getting a batch that the
    # worker is getting an element for while __next__ constantly saves future batches with no limit
    # (which also causes memory usage issues).
    def __init__(self, workers, randomize, dataset, sampleCount, desyncLimit=10):

        super().__init__()

        self.randomize   = randomize
        self.dataset     = dataset
        # The queue is long enough so that prefetching can occur to prevent data loading
        # bottlenecks, an idea from [3e1f51][f832fc] and
        # [448261, torch.utils.data.DataLoader]. Runaway memory usage was also likely a rationale for them, which
        # is why there is a limit on how long the queue can be.
        self.dataQueue   = multiprocessing.Queue(120)
        # This attribute keeps track of the position along self.indices which needs to be handled
        # by a worker next
        self.at          = multiprocessing.Value("I", lock=False)
        self.at.value    = 0
        self.atLock      = multiprocessing.Lock()
        # Raw batch number (not reset when we are done with the epoch)
        self.batchIndex  = 0
        
        
        # Required attributes[09bb19] for a subclass of
        # chainer.iterator.Iterator; I assumed the values for "epoch", "epoch_detail" and
        # "is_new_epoch" were what you see here, while "previous_epoch_detail" is required by that
        # reference to be set to None at the beginning of this class's use. Also, the value that
        # self.batch_size should be is self-evident.
        ############################################################################################
        #                                                                                          #
        
        self.batch_size            = sampleCount
        self.epoch                 = 0
        self.epoch_detail          = 0.0
        self.previous_epoch_detail = None
        self.is_new_epoch          = False
        
        #                                                                                          #
        ############################################################################################

        
        self.indices         = list(range(len(dataset)))
        if self.randomize:
            random.shuffle(self.indices)
        # Used to communicate a new self.indices to the workers
        self.indexListQueues = [multiprocessing.Queue() for i in range(workers)]

        # This dictionary catches dequeued elements from the dataset that have arrived early (they
        # are for a future iteration). They are put into a list that is keyed (in the dictionary) by
        # an iteration number, and they will be fetched in __next__ when the iteration comes up.
        self.overflow             = {}
        self.desyncLimit          = desyncLimit
        self.overflowBatchNumbers = []
        

        # Setting up the workers
        ############################################################################################
        #                                                                                          #

        def fetch(parent_at, parent_atLock, dataset, parent_dataQueue, parent_indexListQueue):
                print("fetching")

                while True:

                    # Getting a new randomized list of indices to for the workers to get in order
                    if parent_at.value == (len(dataset)):
                        indexList = parent_indexListQueue.get()

                    parent_atLock.acquire()
                    temporaryAt      = parent_at.value
                    parent_at.value += 1
                    parent_atLock.release()

                    # Putting images in the queue in any order with respect to process order; datset
                    # index calculation (the modulo part here) uses the striding inspiration
                    # discussed in the commends for FreeFlowIterator.respectiveBatch(...), but with
                    # the major and minor axes equivalents being "dataset" and the remainder
                    # calculated here, respectively.
                    parent_dataQueue.put(  (temporaryAt, dataset[temporaryAt % len(dataset)])  )

        
        # Attribute name same as that of the parameter, which is the same as the initializer
        # parameter for [3e1f51][f832fc]
        self.workers  = [   multiprocessing.Process(
                              target=fetch,
                              args=(self.at,
                                    self.atLock,
                                    self.dataset,
                                    self.dataQueue,
                                    self.indexListQueues[x]),
                              name="counting"
                            )                           for x in range(workers)  ]

        for p in self.workers:
            p.start()

        #                                                                                          #
        ############################################################################################

            


        
        print("fully initialized")







    # The technique encoded by this function may have been inspired by strides used in arrays with
    # more than one dimension (specifically, dividing the number of data points loaded (akin to the
    # minor dimension) by the batch size (major dimension equivalent)).
    def respectiveBatch(self, fetchNumber):
        
        gap           = len(self.dataset) % self.batch_size
        epochsPassed  = math.floor(  fetchNumber / len(self.dataset)  )

        return epochsPassed  +                                                                     \
               math.floor(    (fetchNumber - (gap * epochsPassed)) / self.batch_size    )
        


    
    def __iter__(self):
        return self



    
    def __next__(self):
    
        
        # As stated in the class comment, the order of the queue elements doesn't matter on a
        # per-batch basis; we just gotta make sure the right elements get into the right batch
        ############################################################################################
        #                                                                                          #
        
        batch          = []
        retrievedBatch = False

        # Getting the pairs that we have already retrieved from the queue for this batch
        if self.batchIndex in self.overflow:
            batch  += self.overflow.get( self.batchIndex )
            del self.overflow[self.batchIndex]



        while not retrievedBatch:

            if len(batch) != self.batch_size:
            
                # If we didn't get all batch data from self.overflow, we need to look at the queue
                ####################################################################################
                #                                                                                  #

                xy           = self.dataQueue.get()
                madeForBatch = self.respectiveBatch(xy[0])
                
                if madeForBatch == self.batchIndex:
                    batch.append(xy[1])
                else:
                    # Not the current batch, save it for later. Kill training if it is getting
                    # out-of-contril.
                    ################################################################################
                    #                                                                              #

                    if not madeForBatch in self.overflow:
                        self.overflow[madeForBatch] = []
                        self.overflowBatchNumbers.append(madeForBatch)
                        self.overflowBatchNumbers   = sorted(self.overflowBatchNumbers)
                    
                    if (self.overflowBatchNumbers[-1] - self.batchIndex) >= self.desyncLimit:
                        raise Exception(  "The workers are too far ahead of the batch assembly in \
                                          FreeFlowIterator.__next__(...); at {0}, got {1}"\
                                          .format(self.batchIndex, self.overflowBatchNumbers[-1])
                                       )

                    self.overflow[madeForBatch].append(xy[1])
                    
                    #                                                                              #
                    ################################################################################

                
                #                                                                                  #
                ####################################################################################

            else:

                retrievedBatch = True

        #                                                                                          #
        ############################################################################################
               

        

        self.is_new_epoch = (self.at == len(self.dataset))

        if self.is_new_epoch:

            self.atLock.acquire()
            self.at                      = 0
            self.atLock.release()

            self.epoch                  += 1
            self.previous_epoch_detail   = self.epoch_detail
            self.epoch_detail            = float(self.epoch)

            # Index shuffle and sending the workers the new indices
            ########################################################################################
            #                                                                                      #

            if self.randomize:
                random.shuffle(self.indices)
            for queue in self.indexListQueues:
                if not queue.empty():
                    print("Possible dead subprocess")
                queue.put(self.indices)
            
            #                                                                                      #
            ########################################################################################


        self.batchIndex  += 1

        return batch





    def serialize(self, x):
        x("epoch_detail", self.epoch_detail)
        x("previous_epoch_detail", self.previous_epoch_detail)
        x("is_new_epoch", self.is_new_epoch)
        x("epoch", self.epoch)
        x("at", self.at)
        x("indices", self.indices)
        x("dataQueue", self.dataQueue)
        x("batchIndex", self.batchIndex)
        x("overflow", self.overflow)
# The available attacks to be used with testbench.py

# ^^^chainer^^^ is the reference for this package
import chainer


# This is the FGSM attack proposed in ^^^explaining^^^. Assumes a Chainer-style loss function
def FGSM(data, truth, network, lossFunction, step):
    
    losses  = lossFunction(truth, data, network, False)

    preStep = chainer.functions.sign(
                chainer.grad(  (losses,), (data,)  )[0]
              )

    return chainer.functions.clip(  data + (preStep * step), 0, 255)
 
# This functions runs the PGD attack (from ^^^towards^^^; following its description in ^^^towards^^^
# and ^^^adversarial^^) ^Same signature, except for numRandomPoints and numFGSMRuns, of FGSM as we
# are going to call FGSM repeatedly (as ^^^towards^^^ says PGD essentially uses repeated FGSM); I
# actually independently thought of using FGSM, but not sure if the aforementioned statement by
# ^^^towards^^^ was subconsciously in my mind, so playing it safe here. As in ^^^towards^^^, the
# infinity norm is used for projection.
def PGD(data, truth, network, lossFunction, step, numRandomPoints, numFGSMRuns, limit):


    package    = data.xp

    result     = data.data
    generator  = data.xp.random.default_rng()
    # As is common when finding out which is the optimal item, starting out the optimal values
    # at negative infinity, and replacing them with better real values later on in the code
    bestLosses = package.array(
                    [ float("-inf") for position in range(data.shape[0]) ],
                    dtype=package.float32
                 )

    for pointIndex in range(numRandomPoints):
        # Another spot where we keep pixel values as integers because of the statements made in
        # ^^^explaining^^^ about cameras returning integral values. Limiting between 0 and 255.
        candidates = generator.integers(
                        package.maximum(data.data - limit, 0),
                        package.minimum(data.data + limit, 255),
                        data.shape,
                        package.int32,
                        True
                     )
        for z in range(numFGSMRuns):
            candidates = FGSM(candidates, truth, network, lossFunction, step)
            # Need to pass in an ndarray^^^numpyndarray^^^, so passing in candidates.data instead of
            # candidates is used; using the .data^^^variabledata^^^ attribute for "candidates" here
            # may have been from somewhere
            candidates = chainer.Variable(
                            package.clip(candidates.data, data.data - limit, data.data + limit)
                         )


        losses     = lossFunction(truth, candidates, network, False)
        result     = package.where(
                        package.expand_dims(losses.data > bestLosses, (1, 2, 3)),
                        candidates.data,
                        result
                     )
        bestLosses = package.maximum(bestLosses, losses.data)
        
    return result
        
        
    
                
            
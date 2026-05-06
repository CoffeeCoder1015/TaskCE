from feature.formula import And, Not, Or


def IoU(v1,v2):
    n = ( v1&v2 ).sum()
    d = ( v1|v2 ).sum()
    if d == 0:
        return d
    return n/d


def append_sample(samples, iou, feature_tracker, beam_size):
    samples.append((iou, feature_tracker.flatten()))
    samples.sort(key=lambda x: x[0], reverse=True)
    del samples[beam_size:]


def beamsearch_all(feature_vectors,activation_vectors, beam_size=5, formula_length=5):
    acts_shape = activation_vectors.shape # [examples x neuron number]
    for n in range(acts_shape[1]):
        neuron = activation_vectors[:,n]
        
        best_iou = -float("inf")
        samples = []

        # Built axiom tokens (so as to not extend the formula with tokens with low IoU)
        basis = []
        for feature in feature_vectors:
            iou = IoU(feature[1],neuron)
            basis.append(( iou,feature ))
            if iou > best_iou:
                best_iou = iou
            append_sample(samples, iou, feature[0], beam_size)
        basis.sort(key=lambda x: x[0],reverse=True)
        basis = basis[:beam_size]
        
        # Beam search
        queue = basis.copy()
        new_queue = []
        i = 0
        while i < formula_length:
            _, f= queue.pop()
            feature_tracker, binary_vector = f

            for _,feature in basis:
                # AND
                anded = And(left=feature_tracker,right=feature[0])
                anded_bin = binary_vector & feature[1]
                and_feat = (anded,anded_bin)
                # OR
                ored = Or(left=feature_tracker,right=feature[0])
                ored_bin = binary_vector | feature[1]
                or_feat = (ored,ored_bin)
                # AND_NOT
                anoted = And(left=feature_tracker,right=Not(feature[0]))
                anot_bin = binary_vector & ( ~feature[1] )
                anot_feat = (anoted,anot_bin)
                new_queue.extend([and_feat,or_feat,anot_feat])

            if not queue:
                for feature in new_queue:
                    iou = IoU(feature[1],neuron)
                    queue.append(( iou,feature ))
                    if iou > best_iou:
                        best_iou = iou
                    append_sample(samples, iou, feature[0], beam_size)
                queue.sort(key=lambda x: x[0],reverse=True)
                queue = queue[:beam_size]
                new_queue = []
                i+=1

        exit() # Single iteration for testing

        
        

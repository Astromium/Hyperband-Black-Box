import numpy as np
def calculate_sr(model, labels, scores, checkpoints):
    sr = 0
    adversaries = []
    for i in range(labels.shape[0]):
        cf_idx = np.argmin(scores[i])
        adv = checkpoints[i][cf_idx]
        adversaries.append(adv)
        pred = model.predict(adv)[0]
        y = labels[i]
        #print(f'label, pred for example {i} : {y}, {pred}')
        if pred != y:
            sr += 1
    return round(sr / labels.shape[0], 2), np.array(adversaries)
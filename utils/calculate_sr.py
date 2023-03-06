import numpy as np
from scipy.special import softmax

def calculate_sr_url_rf(model, data, labels, scores, checkpoints):
    sr = 0
    ct = 0
    for i in range(labels.shape[0]):
        if model.predict(data[i][np.newaxis, :])[0] != labels[i]:
            continue
        ct += 1
        cf_idx = np.argmin(scores[i])
        adv = checkpoints[i][cf_idx]
        pred = model.predict(adv[np.newaxis, :])[0]
        y = labels[i]
        #print(f'label, pred for example {i} : {y}, {pred}')
        if pred != y:
            sr += 1
    return round(sr / labels.shape[0], 2)

def calculate_sr_botnet(model, data, labels, scores, checkpoints):
    sr = 0
    ct = 0
    for i in range(labels.shape[0]):
        if np.argmax(softmax(model.predict(data[i].reshape(1, -1)))) != labels[i]:
            print('inside the if')
            continue
        else:
            ct += 1
            cf_idx = np.argmin(scores[i])
            adv = checkpoints[i][cf_idx]
            pred = np.argmax(softmax(model.predict(adv[np.newaxis, :])))
            print(f'cf_idx {cf_idx}')
            print(f'Example {i}. label {labels[i]}, pred {pred} -> adversarial {pred != labels[i]}')
            y = labels[i]
            if pred != y:
                sr += 1
    return sr / ct
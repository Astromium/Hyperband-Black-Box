import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mlc.datasets.dataset_factory import get_dataset
from hyperband import Hyperband
from utils.calculate_sr import calculate_sr
from utils.config_sampler import config_sampler
from utils.generate_perturbation import generate_perturbation
from scipy.special import softmax
from ml_wrappers import wrap_model
import timeit

# load the botnet dataset
ds = get_dataset('ctu_13_neris')
X, y = ds.get_x_y()
metadata = ds.get_metadata()

# scaling 
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_test, y_test = X[143046:], y[143046:]

print('y_test example', y_test[0])

# load the model
model = tf.keras.models.load_model('./ressources/model_botnet.h5')
wrapped_model = wrap_model(model, X_test, model_task='classification')

# get the mutable features
mutables = metadata.index[metadata['mutable'] == True].tolist()

#print('mutables', mutables)

def scoring_func(clf, config, budget, x_clean, y_clean, eps, distance):
     
    score = 0.0
    adv = np.array(x_clean)
    for _ in range(budget):
        perturbation = generate_perturbation(shape=np.array(config).shape, epsilon=eps, distance=distance)
        adv[0][list(config)] += perturbation
        
        pred = softmax(clf.predict(adv))
        score += pred[0][y_clean]
        
    return round(score / budget, 3), adv

BATCH_SIZE = 64
eps = 0.2
distance = 'l2'

dimensions = X_test.shape[1]

scores, checkpoints, configs = [], [], []

if __name__ == "__main__":
    start = timeit.default_timer()
    for i in range(BATCH_SIZE):
        hp = Hyperband(
            objective=scoring_func,
            clf=model,
            x_clean=X_test[i].reshape(1, -1),
            y_clean=y_test[i],
            config_sampler=config_sampler,
            eps=eps,
            dimensions=dimensions,
            max_config_size=len(mutables)-1,
            distance=distance,
            downsample=2
        )
    
        all_scores, all_configs, all_checkpoints = hp.run(mutables=mutables)

        scores.append(all_scores)
        configs.append(all_configs)
        checkpoints.append(all_checkpoints)
    end = timeit.default_timer()

    sr, adv = calculate_sr(model=wrapped_model, labels=y_test[:BATCH_SIZE], scores=np.array(scores), checkpoints=checkpoints)

    print(f'Execution time {(end - start) / 60}')
    print(f'Success rate over {BATCH_SIZE} examples {sr * 100}')





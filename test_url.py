import numpy as np
import warnings
import pandas as pd
from utils.generate_perturbation import generate_perturbation
from utils.calculate_sr import calculate_sr
from utils.config_sampler import config_sampler

warnings.filterwarnings(action='ignore')

from hyperband import Hyperband


def scoring_func(clf, config, budget, x_clean, y_clean, eps, distance, min_max_constraints):
    """
    scoring_func is used to evaluate a configuration
    it generates adversarial examples for a configuration and returns the success rate
    Args: 
        - clf : classifier to query and get predictions from
        - config : a configuration from config_sampler
        - budget : number of adversarials to generate
        - x_clean : original clean input
        - y_clean : original label
        - eps : maximum perturbation
        - the previous best perturbation so far
    returns : success rate of the configuration    
    """
    score = 0.0
    adv = np.array(x_clean)
    for _ in range(budget):
        perturbation = generate_perturbation(shape=np.array(config).shape, epsilon=eps, distance=distance)
        adv[0][list(config)] += perturbation

        # clip if min_max_constraints
        if min_max_constraints:
            min_constraints, max_constraints = min_max_constraints[0], min_max_constraints[1]
            adv[0] = np.clip(adv[0], min_constraints, max_constraints)

        
        pred = clf.predict_proba(adv)
        score += pred[0][y_clean]

    return round(score / budget, 3), adv

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

x_clean = np.load("./ressources/baseline_X_test_candidates.npy")[:10]
y_clean = np.load("./ressources/baseline_y_test_candidates.npy")[:10]

model = joblib.load("./ressources/baseline_rf.model")
preprocessing_pipeline = joblib.load("./ressources/baseline_scaler.joblib")
model_pipeline = Pipeline(
        steps=[("preprocessing", preprocessing_pipeline), ("model", model)]
)

metadata = pd.read_csv('./ressources/url_metadata.csv')

# get the min-max values for features
min_constraints = metadata['min'].tolist()[:63]
max_constraints = metadata['max'].tolist()[:63]
min_max_constraints = (min_constraints, max_constraints)

# get mutable features
mutables = metadata.index[metadata['mutable'] == True].tolist()


import timeit

dimensions = x_clean.shape[1]
scores = []
configs = []
checkpoints = []
start = timeit.default_timer()
batch = 30

if __name__ == '__main__':

    distance = 'inf'
    eps = 0.2

    for i in range(batch):
        x = x_clean[i].reshape(1, -1)
        hp = Hyperband(
            objective=scoring_func,
            clf=model_pipeline,
            x_clean=x,
            y_clean=y_clean[i],
            config_sampler=config_sampler,
            eps=eps,
            dimensions=x.shape[1],
            max_config_size=x.shape[1]-1,
            distance=distance,
            downsample=2
        )
    
        all_scores, all_configs, all_checkpoints = hp.run()

        scores.append(all_scores)
        configs.append(all_configs)
        checkpoints.append(all_checkpoints)
    
    end = timeit.default_timer()

 

    success_rate, adversarials = calculate_sr(model=model_pipeline, labels=y_clean[:batch], scores=np.array(scores), checkpoints=checkpoints)

    print(f'execution time = {(end - start) / 60}')
    print(f'success rate = {success_rate * 100}%')
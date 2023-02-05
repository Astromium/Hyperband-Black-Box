import numpy as np

def generate_perturbation(shape, epsilon, distance):
    if distance == 'l2':
        perturbation = np.random.rand(*shape)
        perturbation = (perturbation / np.linalg.norm(perturbation, ord=2)) * epsilon
        return perturbation
    
    elif distance == 'inf':
        perturbation = np.random.rand(*shape)
        perturbation = (perturbation / np.linalg.norm(perturbation, ord=np.inf)) * epsilon
        return perturbation
    
    else:
        raise 'NotImplementedError'
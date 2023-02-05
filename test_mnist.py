import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml_wrappers import wrap_model
from torchvision import datasets
import torchvision.transforms as T
from scipy.special import softmax
import numpy as np
from utils.generate_perturbation import generate_perturbation
from utils.calculate_sr import calculate_sr
from utils.config_sampler import config_sampler
from hyperband import Hyperband
import timeit


train_data = datasets.MNIST(root='data', train='true', download=True, transform=T.ToTensor())
test_data = datasets.MNIST(root='data', train='false', download=True, transform=T.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fcn = nn.Linear(2048, 10)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fcn(x)
        return x

net = Net()
net.load_state_dict(torch.load('./ressources/model_mnist.pth'))

imgs, labels = next(iter(test_loader))

def scoring_func(clf, config, budget, x_clean, y_clean, eps, distance):
    """
    scoring_func is used to evaluate a configuration
    it generates adversarial examples for a configuration and returns the success rate
    Args: 
        - clf : classifier to query and get feedback from
        - config : a configuration from config_sampler
        - budget : number of adversarials to generate
        - x_clean : original clean input
        - y_clean : original label
        - eps : maximum perturbation
    returns : success rate of the configuration    
    """
    score = 0.0
    adv = np.array(x_clean).reshape(1, -1)

    for _ in range(budget):
        perturbation = generate_perturbation(shape=np.array(config).shape, epsilon=eps, distance=distance)
        adv[0][list(config)] += perturbation

        adv_img = np.array(adv).reshape(1, 28, 28)
        adv_img = np.clip(adv_img, 0, 1)
        pred = softmax(clf.predict_proba(adv_img))
        
        score += 2 * pred[0][y_clean] - np.sum(pred[0]) 

    return round(score / budget, 3), adv_img

wrapped_model = wrap_model(net, imgs, model_task='classification')
dimensions = imgs.shape[2] * imgs.shape[3]

distance = 'l2'
eps = 0.2

scores , configs, checkpoints = [], [], []

if __name__ == "__main__":

    start = timeit.default_timer()

    for i in range(imgs.shape[0]):
        hp = Hyperband(
            objective=scoring_func,
            clf=wrapped_model,
            x_clean=imgs[i],
            y_clean=labels[i].item(),
            config_sampler=config_sampler,
            eps=eps,
            dimensions=dimensions,
            max_config_size=dimensions-1,
            distance=distance,
            downsample=2
        )
    
        all_scores, all_configs, all_checkpoints = hp.run()

        scores.append(all_scores)
        configs.append(all_configs)
        checkpoints.append(all_checkpoints)
    
    end = timeit.default_timer()

    lb = np.array([l.item() for l in labels])
    sr, adversaries = calculate_sr(model=wrapped_model, labels=lb, scores=np.array(scores), checkpoints=checkpoints)

    print(f'Execution time {(end - start) / 60}')
    print(f'Success rate over {imgs.shape[0]} examples: {sr * 100}')
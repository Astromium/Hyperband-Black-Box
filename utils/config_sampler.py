import random

def config_sampler(num_dimensions, num_configs, max_config_size, mutable_features=None):
    """
    config_sampler generates random configurations at the begining of every hyperband bracket
    Args: 
        - num_dimensions : the number of features in the dataset
        - num_configs : the number of configurations we want to generate
        - max_config_size : the maximum number of features each configuration should have
    """
    configs = [None] * num_configs
    if mutable_features:
        sample_list = mutable_features
    else:
        sample_list = list(range(0, num_dimensions))
    for i in range(num_configs):
        n = random.randint(1, max_config_size + 1)
        sample = random.sample(sample_list, n)
        configs[i] = tuple(sample)
    
    return configs
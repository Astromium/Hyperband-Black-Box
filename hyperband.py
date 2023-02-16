from successive_halving import SuccessiveHalving
import math
from tqdm import tqdm
import logging

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='logs.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

class Hyperband():
    def __init__(self, objective, clf, x_clean, y_clean ,config_sampler ,eps ,dimensions ,max_config_size,
              distance='l2',
              max_resources_per_config=81,
              downsample=3,
              total_resources=None,
              random_seed=None):
        
        self.objective = objective
        self.clf = clf
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.config_sampler = config_sampler
        self.eps = eps
        self.dimensions = dimensions
        self.max_config_size = max_config_size
        self.max_resources_per_config = max_resources_per_config
        self.downsample = downsample
        self.total_resources = total_resources
        self.random_seed = random_seed
        self.distance = distance
    
    def run(self, mutables=None, min_max_constraints=None):
        
        if self.downsample <= 1:
            raise ValueError('Downsample must be > 1; otherwise, the number of resources allocated' +
                             'does not grow')

        all_scores = []
        all_configurations = []
        all_checkpoints = []

        # Number of times to run hyperband
        # Ex. `max_resources_per_model = 81 and downsample = 3`
        #     Then => initial_resources = [1, 3, 9, 27, 81]
        #     And => `hyperband_rounds = 5`
        #     And => `successive_halving_rounds = [5, 4, 3, 2, 1]`
        n_hyperband_rounds = math.floor(math.log(self.max_resources_per_config, self.downsample)) + 1
        if self.total_resources is None:
            # the budget B in Hyperband paper
            total_resources_per_round = self.max_resources_per_config * n_hyperband_rounds
        else:
            total_resources_per_round = self.total_resources / n_hyperband_rounds
        total_configurations_evaluated = 0

        # if progress_bar:
        progress_bar = tqdm(total=total_resources_per_round * n_hyperband_rounds)
        setattr(progress_bar, 'stats', {'min_proba': math.inf, 'configurations_evaluated': 0})

        for i in reversed(range(n_hyperband_rounds)):
            n_successive_halving_rounds = i + 1
            # calculating the number of configurations to sample for this round
            #n_configurations = downsample * total_resources_per_round
            #n_configurations /= downsample * (1 + i) - i
            #n_configurations /= downsample**(-i + n_hyperband_rounds - 1)
            n_configurations = (total_resources_per_round * self.downsample ** i) / (self.max_resources_per_config * (i+1))
            n_configurations = round(n_configurations)
            total_configurations_evaluated += n_configurations
            
            round_budget = round(self.max_resources_per_config / self.downsample**i)

            print('Running Hyperband round: ' + str(n_successive_halving_rounds) + 
                  ' with a number of configurations: ' + str(n_configurations)
                  + ' and a budget: ' + str(round_budget)
                 )
            # starting successive halving for the round (bracket)
            min_budget = 5
            if round_budget > 5:
                min_budget = round_budget
                
            sh = SuccessiveHalving(objective=self.objective, clf=self.clf, x_clean=self.x_clean, y_clean=self.y_clean,
                        config_sampler=self.config_sampler,
                        eps = self.eps,
                        dimensions=self.dimensions,
                        max_config_size=self.max_config_size,
                        max_resources_per_configuration=self.max_resources_per_config,
                        distance = self.distance,
                        downsample=self.downsample,
                        initial_resources=min_budget,
                        n_configurations=n_configurations,
                        random_seed=self.random_seed,
                        progress_bar=progress_bar,
                )
            scores, configurations, checkpoints = sh.run(mutables=mutables, min_max_constraints=min_max_constraints)
            
            logger.info('Finished hyperband round: %d of %d', n_hyperband_rounds - i - 1,
                        n_hyperband_rounds - 1)
            logger.info('Scores for round: %s', str(scores))
            logger.info('Configurations for round: %s', str(configurations))
            all_scores.extend(scores)
            all_configurations.extend(configurations)
            all_checkpoints.extend(checkpoints)

        #if isinstance(progress_bar, tqdm):
        progress_bar.close()

        logger.info('Total models evaluated: %f', total_configurations_evaluated)
        logger.info('Total resources used: %f', total_resources_per_round * n_hyperband_rounds)
        logger.info('Total resources used per configuration on average: %f',
                    total_configurations_evaluated / total_resources_per_round * n_hyperband_rounds)

        return all_scores, all_configurations, all_checkpoints
        
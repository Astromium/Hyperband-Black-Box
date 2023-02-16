import math
from tqdm import tqdm


class SuccessiveHalving():
    def __init__(self, objective,clf,config_sampler, x_clean,y_clean,eps,dimensions,max_config_size, distance,max_resources_per_configuration=81,
        downsample=3,
        initial_resources=3,
        n_configurations=45,
        random_seed=None,
        progress_bar=None):
        
        self.objective = objective
        self.clf = clf
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.eps = eps
        self.dimensions = dimensions
        self.max_config_size = max_config_size
        self.max_resources_per_configuration = max_resources_per_configuration
        self.downsample = downsample
        self.initial_resources = initial_resources
        self.n_configurations = n_configurations
        self.random_seed = random_seed
        self.progress_bar = progress_bar
        self.config_sampler = config_sampler
        self.distance = distance
        
    def run(self, mutables, min_max_constraints):
        if self.downsample <= 1:
            raise ValueError('Downsample must be > 1; otherwise, the number of resources allocated' +
                         'does not grow')
            
        round_n_configurations = lambda n: max(round(n), 1)
        total_resources_per_model = 0
        configurations = self.config_sampler(
                self.dimensions, 
                round_n_configurations(self.n_configurations), 
                self.max_config_size,
                mutable_features=mutables
        )
        checkpoints = [[] for _ in range(round_n_configurations(self.n_configurations))]
        scores = [math.inf for _ in range(round_n_configurations(self.n_configurations))]
            
        remember_to_close = False
        #if not isinstance(self.pb_obj, tqdm) and self.progress_bar:
        remember_to_close = True
                # TODO: Compute the tqdm total
        progress_bar = tqdm()
                # Keep tabs on a set of stats
        setattr(progress_bar, 'stats', {'min_proba': math.inf, 'configurations_evaluated': 0})
        i = 0   
        while total_resources_per_model < self.max_resources_per_configuration:
                # Compute number of resources to continue running each model with
            if total_resources_per_model == 0:
                budget = self.initial_resources
            else:
                    #budget = min(
                        #total_resources_per_model * downsample - total_resources_per_model,
                        #max_resources_per_configuration - total_resources_per_model)
                budget = min(
                        total_resources_per_model * self.downsample,
                        self.max_resources_per_configuration
                    )
                    #update_n_resources = total_resources_per_model * downsample - total_resources_per_model    
            results = []
            print(f"Budget is {budget}")

            for score, checkpoint, config in zip(scores, checkpoints, configurations):
                new_score, new_checkpoint = self.objective(self.clf, config, budget, self.x_clean, self.y_clean, self.eps, self.distance, min_max_constraints)
                if new_score < score:
                        #new_score = min(score, new_score)
                    results.append(tuple([new_score, new_checkpoint]))
                else:
                    results.append(tuple([score, checkpoint]))
                #if isinstance(self.pb_obj, tqdm):
                progress_bar.update(budget)
                if progress_bar.stats['min_proba'] > new_score:
                    progress_bar.stats['min_proba'] = new_score
                            #progress_bar.setpostfix(progress_bar.stats)

            total_resources_per_model += budget

                # NOTE: If this is not the last
            is_last_iteration = total_resources_per_model >= self.max_resources_per_configuration
            if not is_last_iteration:
                    # Sort by minimum score `k[0][0]`
                results = sorted(zip(results, configurations), key=lambda k: k[0][0])
                    #print(f'results {results}')
                    #print('initial results ' + str(results))
                configurations_evaluated = len(results) - round_n_configurations(self.n_configurations / self.downsample)
                results = results[:round_n_configurations(self.n_configurations / self.downsample)]
                    #print('results for next round ' + str(results))
                    # Update `hyperparameters` lists
                results, configurations = zip(*results)
                self.n_configurations = self.n_configurations / self.downsample
            else:
                configurations_evaluated = len(results)
            i += 1
            print(f'finished round {i} of successive halving')

                # Update `scores` and `checkpoints` lists
            scores, checkpoints = zip(*results)

                # for testing 
                #print('update_n_ressources = ', update_n_resources)
                #print('total_resources_per_model = ', total_resources_per_model)
                #print('max_resources_per_model = ', max_resources_per_model)

            #if isinstance(progress_bar, tqdm):
            progress_bar.stats['configurations_evaluated'] += configurations_evaluated
            progress_bar.set_postfix(progress_bar.stats)

        if remember_to_close:
            progress_bar.close()

        return scores, configurations, checkpoints
        
        
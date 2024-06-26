from hmmlearn.base import BaseHMM

import numpy as np
import pandas as pd
from scipy.stats import gamma
from sklearn.utils import check_random_state

class BaseGammaHMM(BaseHMM):

    def _get_n_fit_scalars_per_param(self):
        n_comp = self.n_components
        n_feat = self.n_features
        return{
            's' : n_comp - 1,
            't' : n_comp * (n_comp-1),
            'a' : n_comp * n_feat,
            'b' : n_comp * n_feat
        }
    
    def _compute_log_likelihood(self, X):
        logprob = np.empty((len(X), self.n_components))

        for c in range(self.n_components):
            logprob[:, c] = np.sum(gamma.logpdf(X, self.shape_[c], scale=self.scale_[c]), axis=1)

        return logprob
    
    def _initialize_sufficient_statistics(self):
            
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['sum_obs'] = np.zeros((self.n_components, self.n_features))
        stats['prod_obs'] = np.zeros((self.n_components, self.n_features))
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                            posteriors, fwdlattice, bwdlattice):
            
            super()._accumulate_sufficient_statistics(
                stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
    
            print(posteriors)
            stats['post'] += posteriors.sum(axis=0)
            stats['sum_obs'] += np.sum(X, axis=0)
            stats['prod_obs'] = stats['prod_obs'] * np.prod(X, axis=0)

    def _generate_sample_from_state(self, state, random_state):
        random_state = check_random_state(random_state)
        return random_state.gamma(self.shape_[state], scale=self.scale_[state])


class GammaHMM(BaseGammaHMM):

    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0, sumobs_prior=1.0,
                    prodobs_prior=1.0, implementation='log',
                    algorithm='viterbi', random_state=None, n_iter=10, tol=1e-2,
                    verbose=False, params='stab', init_params='stab'):
        
        super().__init__(n_components=n_components, startprob_prior=startprob_prior,
                            transmat_prior=transmat_prior,
                            algorithm=algorithm, random_state=random_state, n_iter=n_iter,
                            tol=tol, verbose=verbose, params=params, init_params=init_params, implementation=implementation)
        
        self.sumobs_prior = sumobs_prior
        self.prodobs_prior = prodobs_prior

    def _init(self, X, lengths=None):
        super()._init(X, lengths=lengths)

        self.random_state = check_random_state(self.random_state)

        mean_X=X.mean(axis=0)
        var_X=X.var(axis=0)

        self.shape_ = self.random_state.gamma(
            shape=mean_X**2 / var_X, 
            scale=var_X / mean_X,
            size=(self.n_components, self.n_features)
        )

        self.scale_ = self.random_state.gamma(
            shape=mean_X**2 / var_X,
            scale=var_X / mean_X,
            size=(self.n_components, self.n_features)
        )

    def _check(self):
        super()._check()

        self.sumobs_prior = np.asarray(self.sumobs_prior, dtype=float)
        self.prodobs_prior = np.asarray(self.prodobs_prior, dtype=float)

        # if self.sumobs_prior.shape != (self.n_components, self.n_features):
        #     raise ValueError('sumobs_prior must have shape (n_components, n_features)')

        # if self.prodobs_prior.shape != (self.n_components, self.n_features):
        #     raise ValueError('prodobs_prior must have shape (n_components, n_features)')
        
    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        sumobs = stats['sum_obs']
        prodobs = stats['prod_obs']
        post = stats['post']


        self.shape_ = (sumobs + self.sumobs_prior) / (post + 1)
        self.scale_ = (prodobs + self.prodobs_prior) / (post + 1)
    

if __name__=='__main__':
     

    something = GammaHMM(n_components=2, random_state=42)
    # print(something)

    # hulls_df=pd.read_csv("data/hulls_df.csv")

    # X = hulls_df["HomeHull"].to_numpy().reshape(-1, 1)
    # Y = hulls_df["AwayHull"].to_numpy().reshape(-1, 1)

    # X = (X - np.mean(X)) / np.std(X)
    # Y = (Y - np.mean(Y)) / np.std(Y)

    # np.random.seed(0)
    random_state = check_random_state(0)
    n_samples = 1000
    n_features = 1
    n_components = 2
    X = np.concatenate([
        gamma.rvs(a = 1, scale=3, size=(n_samples // 2, n_features)),
        gamma.rvs(a = 3, scale=4, size=(n_samples // 2, n_features))
    ])
    # Shuffle X
    X = X[random_state.permutation(n_samples)]
    lengths = [n_samples]

    X = X.reshape(-1, 1)

    _ = something.fit(X)

    print('done')
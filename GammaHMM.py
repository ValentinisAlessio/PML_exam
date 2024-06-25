from hmmlearn.base import BaseHMM, _AbstractHMM
from hmmlearn.stats import log_multivariate_normal_density
from hmmlearn.utils import fill_covars, log_normalize

import numpy as np
from scipy.special import logsumexp
from scipy.stats import gamma
from sklearn.utils import check_random_state

import matplotlib.pyplot as plt
import pandas as pd

class BaseGammaHMM(BaseHMM):

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return{
            's' : nc - 1,
            't' : nc * (nc-1),
            'a' : nc * nf,
            'b' : nc * nf
        }
    
    # def _compute_likelihood(self, X):
    #     probs = np.empty((len(X), self.n_components))
    #     for c in range(self.n_components):
    #         for f in range(self.n_features):
    #             probs[:, c] += gamma.pdf(X[:, f], self.shape_[c, f], scale=self.scale_[c, f])

    #     return probs
    
    def _compute_log_likelihood(self, X):
        logprob = np.empty((len(X), self.n_components))

        for c in range(self.n_components):
            logprob[:, c] = np.sum(gamma.logpdf(X, self.shape_[c], scale=self.scale_[c]), axis=1)

        return logprob
    
    def _initialize_sufficient_statistics(self):

        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['log_obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob, 
                                          posteriors, fwdlattice, bwdlattice):
        
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'a' in self.params or 'b' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, X)
            stats['log_obs'] += np.dot(posteriors.T, np.log(X))

        # if 'b' in self.params:
        #     for c in range(self.n_components):      # Non sono convinto sul secondo for
        #         for f in range(self.n_features):
        #             stats['obs'][c, f] += np.sum(posteriors[:, c] * X[:, f])

    def _generate_sample_from_state(self, state, random_state):
        random_state = check_random_state(random_state)
        return random_state.gamma(self.shape_[state], scale=self.scale_[state])
    
    # def _do_mstep(self, stats):
    #     super()._do_mstep(stats)
    #     nc = self.n_components
    #     nf = self.n_features

    #     if 'a' in self.params or 'b' in self.params:
    #         post = stats['post']
    #         obs = stats['obs']
    #         log_obs = stats['log_obs']

    #         self.shape_ = post / (post * (np.log(post) - log_obs / post) - obs / post)
    #         self.scale_ = obs / (post * self.shape_)

class GammaHMM(BaseGammaHMM):

    def __init__(self, n_components=1, algorithm='viterbi', startprob_prior=1.0,
                 transmat_prior=1.0, n_iter=10,
                    tol=1e-2, verbose=False, params='stab', init_params='stab',
                    random_state=None, implementation='log', scale_prior=1.0,
                    shape_prior=1.0):
        BaseHMM.__init__(self, n_components,
                            startprob_prior=startprob_prior,
                            transmat_prior=transmat_prior,
                            algorithm=algorithm,
                            random_state=random_state,
                            n_iter=n_iter, tol=tol, verbose=verbose,
                            params=params, init_params=init_params,
                            implementation=implementation)
                         
        self.scale_prior = scale_prior
        self.shape_prior = shape_prior
        self.shape_ = None
        self.scale_ = None

    def _init(self, X, lengths=None):
        super()._init(X, lengths=lengths)

        self.random_state = check_random_state(self.random_state)

        # self.shape_ = np.ones((self.n_components, self.n_features))
        # self.scale_ = np.ones((self.n_components, self.n_features))
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

        print(self.shape_, self.scale_)

    def _check(self):
        super()._check()

        self.shape_ = np.asarray(self.shape_)
        self.scale_ = np.asarray(self.scale_)

        if self.shape_.shape != (self.n_components, self.n_features):
            raise ValueError("shape_ must have shape (n_components, n_features)")

        if self.scale_.shape != (self.n_components, self.n_features):
            raise ValueError("scale_ must have shape (n_components, n_features)")
        
        self.n_features = self.shape_.shape[1]

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        if 'a' in self.params or 'b' in self.params:
            post = stats['post']
            obs = stats['obs']
            log_obs = stats['log_obs']

            # IL PROBLEMA È QUA!!

            self.shape_ = post / (post * (np.log(post) - log_obs / post) - obs / post)
            self.scale_ = obs / (post * self.shape_)
                                                  


if __name__ == "__main__":

    # Generate a random sequence of gamma-distributed data from a 2-state HMM
    # np.random.seed(0)
    # random_state = check_random_state(0)
    # n_samples = 1000
    # n_features = 2
    # n_components = 2
    # X = np.concatenate([
    #     gamma.rvs(a = 1, scale=3, size=(n_samples // 2, n_features)),
    #     gamma.rvs(a = 3, scale=4, size=(n_samples // 2, n_features))
    # ])
    # # Shuffle X
    # X = X[random_state.permutation(n_samples)]
    # lengths = [n_samples]

    hulls_df=pd.read_csv("data/hulls_df.csv")

    X = hulls_df["HomeHull"].to_numpy()
    Y = hulls_df["AwayHull"].to_numpy()

    print(X)

    # Fit the model
    model = GammaHMM(n_components=2, n_iter=100)
    print("GammaHMM.py loaded")
    model.fit(X.reshape(-1, 1))

    print("\nGaussian distribution means:")
    print(model.shape_)

    print("\nGaussian distribution covariances:")
    print(model.scale_)

    print("\nStart probabilities:")
    print(np.round(model.startprob_, 2))

    print("\nTransition matrix:")
    print(np.round(model.transmat_,2))

    _ = plt.imshow(model.transmat_)
    _ = plt.colorbar()

    # basemodel = BaseGammaHMM(n_components=n_components)
    # basemodel._compute_log_likelihood(X)

    print("Done!")
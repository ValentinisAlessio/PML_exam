import torch
import pyro
import pyro.distributions as dist
from pyro import poutine

def UnivariateHMM(sequence: torch.tensor, 
                  hidden_dim: int, 
                  include_prior: bool=True):
    '''
    Pyro Model for a Hidden Markov Model with a single univariate observation with Gamma emission distribution.
    Structure of the model taken from the Pyro documentation:
    https://pyro.ai/examples/hmm.html
    
    INPUTS:
    - sequence (torch.tensor): A 1-dimensional tensor of observations.
    - hidden_dim (int): The number of hidden states.
    - include_prior (bool): If True, include priors for the parameters of the model.
    '''
    length = len(sequence)
    with poutine.mask(mask=include_prior):
        # Transition probabilities
        probs_x = pyro.sample(
            "probs_x",
            dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(),
        )
        # Emission probabilities (1-dimensional for the area)
        probs_alpha = pyro.sample(
            "probs_alpha",
            dist.Gamma(1.0, 1.0).expand([hidden_dim]).to_event(1)
        )

        probs_beta = pyro.sample(
            "probs_beta",
            dist.Gamma(1.0, 1.0).expand([hidden_dim]).to_event(1)
        )
    
    x = 0  # Initial hidden state
    for t in pyro.markov(range(length)):
        x = pyro.sample(
            f"x_{t}",
            dist.Categorical(probs_x[x]),
            infer={"enumerate": "parallel"},
        )
        pyro.sample(
            f"y_{t}",
            dist.Gamma(probs_alpha[x], probs_beta[x]),
            obs=sequence[t],
        )
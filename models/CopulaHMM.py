import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from utils.CopulaHelpers import copulamodel_log_pdf

def CopulaHMM(sequence: torch.tensor, 
              hidden_dim: int, 
              include_prior: bool=True):
    '''
    Pyro Model for a Hidden Markov Model with a bivariate observation with Copula emission distribution.
    Structure of the model taken from the Pyro documentation:
    https://pyro.ai/examples/hmm.html
    
    INPUTS:
    - sequence (torch.tensor): A 2-dimensional tensor of observations.
    - hidden_dim (int): The number of hidden states.
    - include_prior (bool): If True, include priors for the parameters of the model.
    '''
    n_obs = sequence.shape[0]
    with poutine.mask(mask=include_prior):
        #---------------------------------------------------------------------
        # Transition probabilities
        probs_x = pyro.sample(
            "probs_x",
            dist.Dirichlet(0.9 * torch.eye(hidden_dim) + 0.1).to_event(),
        )
        #---------------------------------------------------------------------
        # Prior for the parameters of emission probabilities 
        probs_alpha1 = pyro.sample(
            "probs_alpha1",
            dist.Gamma(concentration=15.0,rate=0.8).expand([hidden_dim]).to_event(1)
        )

        probs_beta1 = pyro.sample(
            "probs_beta1",
            dist.Gamma(concentration=1.0,rate=1.0).expand([hidden_dim]).to_event(1)
        )
        probs_alpha2 = pyro.sample(
            "probs_alpha2",
            dist.Gamma(concentration=15.0,rate=0.8).expand([hidden_dim]).to_event(1)
        )

        probs_beta2 = pyro.sample(
            "probs_beta2",
            dist.Gamma(concentration=1.0,rate=1.0).expand([hidden_dim]).to_event(1)
        )
        #---------------------------------------------------------------------
        # Prior for theta
        theta = pyro.sample(
            "theta",
            dist.Gamma(5.0,0.7).expand([hidden_dim]).to_event(1)
        )
        
    
    x = 0  # Initial hidden state
    for t in pyro.markov(range(n_obs)):
        x = pyro.sample(
            f"x_{t}",
            dist.Categorical(probs_x[x]),
            infer={"enumerate": "parallel"},
        )
        log_pdf = copulamodel_log_pdf(
            x=sequence[t,0],
            y=sequence[t,1],
            shape1=probs_alpha1[x],
            rate1=probs_beta1[x],
            shape2=probs_alpha2[x],
            rate2=probs_beta2[x],
            theta=theta[x]
        )
        pyro.factor(f"xy_{t}", log_pdf)
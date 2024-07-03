import torch
import pyro
import pyro.distributions as dist

def PredictHMM(num_pred : int,
               initial_state: torch.tensor,
               transition_matrix: torch.tensor,
               shape_params1: torch.tensor,
               rate_params1: torch.tensor,
               shape_params2: torch.tensor,
               rate_params2: torch.tensor) -> torch.tensor:
    
    # Predict the sequence of states
    states=torch.tensor([initial_state])
    for p in range(num_pred):
        x = pyro.sample(
            f"x_{p}",
            dist.Categorical(transition_matrix[states[-1],:])
        )
        states=torch.cat((states,torch.tensor([x])))
    
    # Predict the areas
    alpha1=shape_params1[states[1:]]
    alpha2=shape_params2[states[1:]]
    beta1=rate_params1[states[1:]]
    beta2=rate_params2[states[1:]]
    X= pyro.sample(
        f"X",
        dist.Gamma(alpha1, beta1)
    )
    Y= pyro.sample(
        f"Y",
        dist.Gamma(alpha2, beta2)
    )
    areas = torch.stack((X, Y), dim=1)
        
    return states[1:], areas
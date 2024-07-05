import torch
from utils.CopulaHelpers import copulamodel_log_pdf

def Viterbi(observations: torch.tensor, 
            initial_states_prob: torch.tensor, 
            transition_matrix: torch.tensor,
            shape_params1: torch.tensor,
            rate_params1: torch.tensor,
            shape_params2: torch.tensor,
            rate_params2: torch.tensor,
            theta: torch.tensor) -> torch.tensor:
    """Viterbi algorithm to find the most probable state sequence.
    
    INPUTS:
    - observations (torch.tensor): A 2-dimensional tensor with the observations.
    - initial_states_prob (torch.tensor): A 1-dimensional tensor with the initial probabilities of the states.
    - transition_matrix (torch.tensor): A 2-dimensional tensor with the transition probabilities between states.
    - shape_params1 (torch.tensor): A 1-dimensional tensor with the shape parameters of the first gamma distribution.
    - rate_params1 (torch.tensor): A 1-dimensional tensor with the rate parameters of the first gamma distribution.
    - shape_params2 (torch.tensor): A 1-dimensional tensor with the shape parameters of the second gamma distribution.
    - rate_params2 (torch.tensor): A 1-dimensional tensor with the rate parameters of the second gamma distribution.
    - theta (torch.tensor): A 1-dimensional tensor with the theta parameters of the copula.
    
    OUTPUTS:
    - (torch.tensor) Most likely sequence of states.
    """
    num_obs = observations.shape[0]
    num_states = transition_matrix.shape[0]
    
    # Initialize the dynamic programming table
    V = torch.zeros((num_states, num_obs))
    path = torch.zeros((num_states, num_obs), dtype=int)
    
    # Compute the log probabilities for the initial states
    log_initial_states_prob = torch.log(initial_states_prob)
    log_transition_matrix = torch.log(transition_matrix)
    
    # Compute the log pdf for the first observation across all states
    for s in range(num_states):
        V[s, 0] = log_initial_states_prob[s] + copulamodel_log_pdf(x=observations[0, 0],
                                                                   y=observations[0, 1],
                                                                   shape1=shape_params1[s],
                                                                   rate1=rate_params1[s],
                                                                   shape2=shape_params2[s],
                                                                   rate2=rate_params2[s],
                                                                   theta=theta[s])
        path[s, 0] = 0
    
    # Vectorized Viterbi for t > 0
    for t in range(1, num_obs):
        log_probs = torch.empty(num_states, num_states)
        for s in range(num_states):
            log_probs[:, s] = V[:, t-1] + log_transition_matrix[:, s] + copulamodel_log_pdf(x=observations[t, 0],
                                                                                           y=observations[t, 1],
                                                                                           shape1=shape_params1[s],
                                                                                           rate1=rate_params1[s],
                                                                                           shape2=shape_params2[s],
                                                                                           rate2=rate_params2[s],
                                                                                           theta=theta[s])
        V[:, t], path[:, t] = log_probs.max(dim=0)
    
    # Backtrack to find the most probable state sequence
    optimal_path = torch.zeros(num_obs, dtype=int)
    optimal_path[num_obs-1] = torch.argmax(V[:, num_obs-1])
    
    for t in range(num_obs-2, -1, -1):
        optimal_path[t] = path[optimal_path[t+1], t+1]
    
    return optimal_path


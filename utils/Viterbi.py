import torch
from utils.CopulaHelpers import copulamodel_log_pdf

def Viterbi(observations, initial_states_prob, transition_matrix,
            shape_params1,rate_params1,shape_params2,rate_params2,theta):
    """Viterbi algorithm to find the most probable state sequence.
    
    Args:
        observations: List of observations.
        states: List of states.
        start_prob: Initial state probabilities.
        trans_prob: State transition probabilities.
        emit_params: Parameters for the emission probabilities.
    
    Returns:
        Most likely sequence of states.
    """
    num_obs = observations.shape[0]
    num_states = transition_matrix.shape[0]
    
    # Initialize the dynamic programming table
    V = torch.zeros((num_states, num_obs))
    path = torch.zeros((num_states, num_obs), dtype=int)
    
    # Initialize the base cases (t == 0)
    for s in range(num_states):
        V[s, 0] = torch.log(initial_states_prob[s]) + copulamodel_log_pdf(x=observations[0,0],
                                                                          y=observations[0,1],
                                                                          shape1=shape_params1[s],
                                                                          rate1=rate_params1[s],
                                                                          shape2=shape_params2[s],
                                                                          rate2=rate_params2[s],
                                                                          theta=theta[s])
        path[s, 0] = 0
    
    # Run Viterbi for t > 0
    for t in range(1, num_obs):
        for s in range(num_states):
            prob_state = []
            for s_prev in range(num_states):
                prob = V[s_prev, t-1] + torch.log(transition_matrix[s_prev, s]) + copulamodel_log_pdf(x=observations[t,0],
                                                                                                   y=observations[t,1],
                                                                                                   shape1=shape_params1[s],
                                                                                                   rate1=rate_params1[s],
                                                                                                   shape2=shape_params2[s],
                                                                                                   rate2=rate_params2[s],
                                                                                                   theta=theta[s])
                prob_state.append(prob)
                
            V[s, t] = max(prob_state)
            path[s, t] = torch.argmax(torch.tensor(prob_state))
    
    # Backtrack to find the most probable state sequence
    optimal_path = torch.zeros(num_obs, dtype=int)
    optimal_path[num_obs-1] = torch.argmax(V[:, num_obs-1])
    
    for t in range(num_obs-2, -1, -1):
        optimal_path[t] = path[optimal_path[t+1], t+1]
    
    return optimal_path
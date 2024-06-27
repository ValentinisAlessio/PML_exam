import numpy as np
from scipy.stats import gamma
from scipy.optimize import minimize


def _init_params(n_states):
    # trans_mat = np.random.rand((n_states, n_states))
    # prob_mat = np.diag(np.zeros(n_states))
    # initial_prob = np.empty(n_states)
    # Initialize the transition matrix
    trans_mat = np.random.rand(n_states, n_states)
    initial_prob = np.ones(n_states) / n_states
    gamma_params = np.ones((n_states, 2))

    return trans_mat, initial_prob, gamma_params

def _construct_trans_mat(params, n_states):
    trans_mat = np.zeros((n_states, n_states))
    idx = 0
    for i in range(n_states):
        for j in range(n_states -1):
            trans_mat[i, j] = params[idx]
            idx += 1
        
        trans_mat[i, -1] = 1 - trans_mat[i, :-1].sum()

    return trans_mat

def _construct_initial_probs(params):
    initial_probs= np.array(params)
    initial_probs[-1] = 1 - initial_probs[:-1].sum()

    return initial_probs


def _forward(initial_prob, trans_mat, gamma_params, X):
    
    # trans_mat, initial_prob, gamma_params = _init_params(n_states)
    # n_states = initial_prob.shape[0]

    n_samples, n_features = X.shape

    # print(trans_mat, initial_prob, gamma_params)

    _lik = initial_prob * gamma.pdf(X[0], a=gamma_params[:, 0], scale=gamma_params[:, 1])

    # print(_lik)

    for i in range(1, n_samples):
        _lik = _lik @ trans_mat * gamma.pdf(X[i], a=gamma_params[:, 0], scale=gamma_params[:, 1])
        # print(_lik)

    # print(_lik)
    # print(_lik.shape)
    # print(trans_mat.shape)
    # print(gamma_params.shape)
    # print(initial_prob.shape)

    return _lik.sum()

def _compute_neg_log_lik(params, X, n_states):

    shape_params = params[:n_states]
    scale_params = params[n_states:2*n_states]
    initial_prob_params = params[2*n_states:3*n_states]
    trans_mat_params = params[3*n_states:]

    initial_prob = _construct_initial_probs(initial_prob_params)
    trans_mat = _construct_trans_mat(trans_mat_params, n_states)
    gamma_params = np.column_stack((shape_params, scale_params))

    lik = _forward(initial_prob, trans_mat, gamma_params, X)

    return -np.log(lik)

if __name__ == "__main__":
    
    # generate 10 samples from a gamma distribution
    X1 = gamma.rvs(a=2, scale=2, size=(100, 1))
    X2 = gamma.rvs(a=3, scale=1, size=(100, 1))
    X = np.concatenate((X1, X2), axis=0)
    
    random_state = np.random.RandomState(0)
    X = X[random_state.permutation(200)]

    
    n_states = 2

    # params = np.array([0.5, 0.5, 1, 1, 0.5, 0.5, 0.7, 0.4])
    # lik = _compute_neg_log_lik(params, X, n_states)
    # print(lik)

    # minimize log-likelihood in the parameters
    initial_shape_params = np.array([2.5, 3.1])
    initial_scale_params = np.array([2.5, 1.5])
    initial_initial_prob_params = np.random.rand(n_states )
    initial_transitions_params = np.random.rand(n_states * (n_states ))
    initial_params = np.concatenate((initial_shape_params, initial_scale_params, initial_initial_prob_params, initial_transitions_params))

    bounds = [(0, None)] * (2 * n_states) + [(0, 1)] * n_states + [(0, 1)] * (n_states * (n_states))
    
    result = minimize(_compute_neg_log_lik, initial_params, args=(X, n_states), bounds=bounds)

    estimated_shape = result.x[:n_states]
    estimated_scale = result.x[n_states:2*n_states]
    estimated_initial_prob = _construct_initial_probs(result.x[2*n_states:3*n_states])
    estimated_trans_mat = _construct_trans_mat(result.x[3*n_states:], n_states)

    print(estimated_shape, estimated_scale, estimated_initial_prob, estimated_trans_mat)
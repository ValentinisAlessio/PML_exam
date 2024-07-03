import torch
import pyro.distributions as dist

def empirical_gamma_cdf(x, shape, rate):
    # Generate 1000 random samples from the gamma distribution
    # Think about setting a seed to avoid too much stochasticity
    #torch.manual_seed(3407)
    samples = torch.distributions.Gamma(shape, rate).sample((3500,))
    return (samples <= x).float().mean()

def copula_term_log(theta: torch.tensor, u: torch.tensor, v: torch.tensor):
    log_numerator = torch.log(theta) + torch.log(torch.exp(theta) - 1.0) + theta * (1.0 + u + v)
    denominator = (torch.exp(theta) - torch.exp(theta + theta * u) + torch.exp(theta * (u + v)) - torch.exp(theta + theta * v))**2
    log_denominator = torch.log(denominator)
    return log_numerator - log_denominator

def copulamodel_log_pdf(x,y,shape1,rate1,shape2,rate2,theta):
    g1_lpdf= dist.Gamma(shape1,rate1).log_prob(x)
    g2_lpdf= dist.Gamma(shape2,rate2).log_prob(y)
    u= empirical_gamma_cdf(x, shape1, rate1)
    v= empirical_gamma_cdf(y, shape2, rate2)
    # Qui pensare se fare una if su u e v diversi da circa 0...in quel caso non calcolare il copula term (lo lascio nullo)
    lpdf=g1_lpdf+g2_lpdf
    if (torch.abs(u) > 1e-6) & (torch.abs(v) > 1e-6):
        lpdf += copula_term_log(theta=theta,u=u,v=v)
    return lpdf
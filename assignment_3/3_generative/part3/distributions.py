"""
This file contains classes for a bimodal Gaussian distribution and a
multivariate Gaussian distribution with diagonal covariance matrix.

Author: Deep Learning Course, C.Winkler | Fall 2020
Date Created: 2020-11-25
"""

import numpy as np
import torch


def broadcast(x, a, b):
    """
    Broadcast shape of input tensors a and b to be able to perform element-wise
    multiplication along the last dimension of x.
    Inputs:
    x - Input tensor of shape [n, n, d].
    a - First input tensor of shape [d].
    b - Second input tensor of shape [d].

    Returns:
        Tensor of shape [1, 1, d]
    """
    return (t.view(((1,) * (len(x.shape)-1)) + x.shape[-1:]) for t in [a, b])


class BimodalGaussianDiag:
    """
    Class specifying a Bimodal Bivariate Gaussian distribution with diagonal
    covariance matrix. Contains functions to compute the log-likelihood and to
    sample from the distribution.

    Inputs:
        mu (list)    - List of tensors of length 2. Each element in the list
                       is of shape of 1xdims. These are the mean values of a
                       distribution for each random variable.
        sigma (list) - List of tensors of length 2. Each element in the list is
                       of shape 1xdims. These are the values of standard
                       devations of a distribution for each random variable.
        dims(int)    - Dimensionality of random vector.
    """
    def __init__(self, mu, sigma, dims):
        # TODO: Implement initalization
        self.p1 = MultivariateGaussianDiag(mu[0], sigma[0], dims)
        self.p2 = MultivariateGaussianDiag(mu[1], sigma[1], dims)
        self.mus = mu
        self.sigmas = sigma
        self.dims = dims
        

    def log_prob(self, x):
        # TODO: Implement log probability computation
        p1 = 0.5 * self.p1.log_prob(x).exp()
        p2 = 0.5 * self.p2.log_prob(x).exp()
        logp = torch.log(p1 + p2)
        return logp

    def sample(self, num_samples):
        # TODO: Implement sampling procedure
        samples = torch.zeros([num_samples, 2])
        idx = torch.randint(0, 2, [num_samples], dtype=torch.bool)
        samples[idx] = self.p1.sample([num_samples])[idx]
        samples[~idx] = self.p2.sample([num_samples])[~idx]
        return samples


class MultivariateGaussianDiag:
    """
    Class specifying a Multivariate Gaussian distribution with diagonal
    covariance matrix. Contains functions to compute the log-likelihood and
    sample from the distribution.

    Inputs:
        mu (tensor)    - Tensor of shape of 1xdims. These are
                         the mean values of the distribution for each
                         random variable.
        sigma (tensor) - Tensor of shape 1xdims. These are the
                         values of standard devations of each random variable.
        dims(int)    - Dimensionality of random vector.
    """
    def __init__(self, mu, sigma, dims):
        super().__init__()
        # TODO: Implement initalization
        self.mus = mu
        self.sigmas = sigma
        self.dims = dims

    def log_prob(self, x):
        # TODO: Implement log probability computation
        mu, sigma = list(broadcast(x, self.mus, self.sigmas))
        res = (x - mu)**2/sigma
        log_p = - 0.5*self.dims*np.log(2*np.pi) - 0.*self.dims*np.log(torch.det(torch.diag(self.sigmas)))-1*self.dims*(res).sum(-1)
        return log_p

    def sample(self, num_samples):
        # TODO: Implement sampling procedure
        m = torch.distributions.MultivariateNormal(self.mus, torch.diag(self.sigmas))
        samples = m.sample(torch.tensor([num_samples]))
        return samples

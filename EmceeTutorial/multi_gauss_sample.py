import numpy as np
import matplotlib.pyplot as plt
import emcee

def log_prop(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

ndim = 10
np.random.seed(42)
means = np.random.rand(ndim)
cov = 0.5 - np.random.rand(ndim, ndim)
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)

nwalkers = 250
p0 = np.random.rand(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prop, args=(means, cov))

pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(pos, 1000)

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

for i in range(ndim):
    plt.hist(sampler.flatchain[:,i], 100, histtype="step", label="Dimension {0:d}".format(i))

plt.legend()
plt.title("Multivariant Gaussian Distribution")
plt.show()

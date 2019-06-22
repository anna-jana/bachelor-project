import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import emcee
import corner

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534
print("m_true =", m_true, "b_true =", b_true, "f_true =", f_true)

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# linear least squares fit
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
print("m_ls =", m_ls, "b_ls =", b_ls) # , "f_ls =", f_ls)

# maximum likelihood
def neg_ln_like(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2*lnf))
    return -0.5 * (np.sum((y - model)**2 * inv_sigma2 - np.log(inv_sigma2)))

res = opt.minimize(lambda *args: -neg_ln_like(*args), [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = res["x"]
f_ml = np.exp(lnf_ml)
print("m_ml =", m_ml, "b_ml =", b_ml, "f_ml =", f_ml)

# uncertainty estimation
def lnprior(theta):
    m, b, lnf = theta
    if -5 < m < 0.5 and 0 < b < 10 and -10 < lnf < 1:
        return 0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + neg_ln_like(theta, x, y, yerr)

ndim = 3
nwalkers = 100
pos = [res["x"] + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
sampler.run_mcmc(pos, 1000)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# chain: (num_walker, steps, n_dim = #parameters)

s = samples.copy()
s[:, 2] = np.exp(s[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(s, [16, 50, 84],
                                                axis=0)))
print("m_mcmc =", m_mcmc, "b_mcmc =", b_mcmc, "f_mcmc =", f_mcmc)

# plot
plt.figure(1)
plt.errorbar(x, y, yerr=yerr, fmt=".", color="k", label="data")
plt.plot(x, m_true * x + b_true, color="k", label="true")
plt.plot(x, m_ls * x + b_ls, "--k", label="linear least square fit")
plt.plot(x, m_ml * x + b_ml, ":k", label="maximum likelihood")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.figure(2)
plt.subplot(3,1,1)
plt.ylabel("m")
plt.axhline(m_true)
plt.plot(sampler.chain[:, :, 0].T, "k", linewidth=0.1, alpha=0.8)

plt.subplot(3,1,2)
plt.ylabel("b")
plt.axhline(b_true)
plt.plot(sampler.chain[:, :, 1].T, "k", linewidth=0.1, alpha=0.8)

plt.subplot(3,1,3)
plt.ylabel("f")
plt.xlabel("step number")
plt.axhline(f_true)
plt.plot(np.exp(sampler.chain[:, :, 2].T), "k", linewidth=0.1, alpha=0.8)

plt.figure(3)
xl = np.array([0, 10])
for i, (m, b, lnf) in enumerate(samples[np.random.randint(len(samples), size=100)]):
    if i == 0:
        plt.plot(xl, m*xl + b, color="k", alpha=0.1, label="sample fits")
    else:
        plt.plot(xl, m*xl + b, color="k", alpha=0.1)
plt.plot(xl, m_true * xl + b_true, color="r", lw=2, alpha=0.8, label="true")
plt.errorbar(x, y, yerr=yerr, fmt=".k", label="data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"], truths=[m_true, b_true, np.log(f_true)])

plt.show()

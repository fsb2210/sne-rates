"""
sne-rates
---

Compute supernovae rates using MCMC Bayesian inference
"""

import corner
import numpy as np
import emcee
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt

# data
peer_data = np.array([0.4, 1.5, 2.0]) # survey rates for Sbc galaxies
mw_data = 7                           # 5 historical + 2 remnants
t_mw = 10                             # centuries

def log_prior(theta):
    mu, lam_mw, p = theta
    if mu <= 0 or lam_mw <= 0 or not (0 < p < 1): return -np.inf

    # global average (exponential)
    lp_mu = stats.expon.logpdf(mu, scale=2.0)
    # hierarchical MW (Gamma, linked to mu)
    lp_lam_mw = stats.gamma.logpdf(lam_mw, a=mu, scale=1.0)
    # detection (beta)
    lp_p = stats.beta.logpdf(p, 4, 2)

    return lp_mu + lp_lam_mw + lp_p

def log_likelihood(theta):
    mu, lam_mw, p = theta
    # peer evidence
    ll_peers = np.sum(stats.poisson.logpmf(peer_data, mu))
    # local evidence (corrected for detection p)
    ll_mw = stats.poisson.logpmf(mw_data, lam_mw * p * t_mw)

    return ll_peers + ll_mw

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# MCMC setup
initial = [2.0, 2.0, 0.5]
nwalkers, ndim = 32, 3
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 50000, progress=True)

samples = sampler.get_chain(discard=500, flat=True)

# extract samples from the sampler
# samples[:, 0] is mu (global average)
# samples[:, 1] is lam_mw (Milky Way specific)
mu_samples = samples[:, 0]
lam_mw_samples = samples[:, 1]

# make graphics!
plt.figure(figsize=(10, 6))
sns.kdeplot(mu_samples, label=r"Global Average ($\mu$)", fill=True, color="royalblue", alpha=0.4)
sns.kdeplot(lam_mw_samples, label=r"Milky Way Rate ($\lambda_{mw}$)", fill=True, color="darkorange", alpha=0.4)

plt.axvline(2.0, color='black', linestyle='--', label="Classic '2 per century' rule")

plt.title("Is the Milky Way Typical? \nPosterior Distributions of Supernova Rates")
plt.xlabel("Supernovae per Century")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.savefig("posteriors.png")

# define labels for the corner plot
labels = [r"$\mu$ (Global)", r"$\lambda_{mw}$ (Local)", r"$p$ (Detection)"]
fig = corner.corner(samples, labels=labels, truths=[None, 2.0, None],
                    show_titles=True, title_fmt=".2f", color="darkslateblue")

plt.savefig("corner.png")

"""
sne-rates
---

Compute supernovae rates using MCMC Bayesian inference
"""
from operator import truth

from multiprocessing import Pool

import corner
import numpy as np
import emcee
import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt

# data
k_obs = 7  # 5 historical + 2 remnants
t_obs = 10

def log_probability(theta):
    lam, p = theta
    # Boundary constraints
    if not (0 < lam < 5 and 0.01 < p < 1):
        return -np.inf

    # Prior: We know visibility (p) is low, likely around 20%
    lp = stats.beta.logpdf(p, 2, 8) 

    # Likelihood: The Poisson count
    ll = stats.poisson.logpmf(k_obs, lam * p * t_obs)

    return lp + ll

# MCMC setup
initial = [2.0, 0.5]
nwalkers, ndim = 32, 2
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    sampler.run_mcmc(pos, 2000, progress=True)

samples = sampler.get_chain(discard=500, flat=True)

# extract samples from the sampler
# samples[:, 0] is mu (global average)
# samples[:, 1] is lam_mw (Milky Way specific)
lam_mw_samples = samples[:, 0]
p_samples = samples[:, 1]

# forecast
for yrs in [10, 25, 50]:
    t = yrs / 100
    # probability of at least one event: 1 - P(zero events)
    prob_dist = 1 - np.exp(-lam_mw_samples * t)
    mean_prob = np.mean(prob_dist) * 100
    print(f"- chance of a supernova in the next {yrs} years: {mean_prob:.1f}%")

# make graphics!
plt.figure(figsize=(10, 6))
sns.kdeplot(lam_mw_samples, label=r"Milky Way rate ($\lambda_{mw}$)", fill=True, color="royalblue", alpha=0.4)
# sns.kdeplot(p_samples, label=r"detection probability", fill=True, color="darkorange", alpha=0.4)
plt.axvline(2.0, color='black', linestyle='--', label="Classic '2 per century' rule")

plt.title("Posterior distributions of supernova rates")
plt.xlabel("# supernovae per century")
plt.ylabel("probability density")
plt.xlim(0, 5)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.savefig("posteriors.png")

# labels for the corner plot
labels = [r"$\lambda$", r"$p$"]
# create the corner plot
fig = corner.corner(samples, labels=labels, truths=[2.0, 0.2], show_titles=True, title_fmt=".2f", color="darkslateblue", smooth=1.0)
plt.savefig("corner.png")

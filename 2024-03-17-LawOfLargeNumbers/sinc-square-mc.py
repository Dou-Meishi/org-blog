import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)

# number of samples
N = 1000000
# stand deviation of the reference distribution
sigma = 10

# define f(x)
def f(x):
    return np.sinc(x / np.pi) ** 2

# define p(x)
def p(x):
    return norm.pdf(x, 0, sigma)

# normal random variables, mean 0 variance 1
x_n = np.random.normal(0, sigma, N)

# compute y_n
y_n = f(x_n) / p(x_n)

# apply the Strong Law of Large Numbers
averages = np.cumsum(y_n) / np.arange(1, N+1)

# plot the averages
plt.figure(figsize=(10,5))
plt.plot(averages, label="Empirical mean")
plt.axhline(y=np.pi, color='r', linestyle='--', label="Expectation")
plt.title('Estimate the integral of sinc square')
plt.xlabel('Number of samples')
plt.ylabel('Estimate of integral')
plt.legend()
plt.savefig("./sinc-square-integral.png")

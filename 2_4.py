import numpy as np 
from math import pi
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_f
from scipy.stats import gamma, norm

def update_mu(mu_0, lam_0, X):
    N = len(X)
    mu_N = (lam_0 * mu_0 + np.sum(X))/(lam_0 + N)
    return mu_N

def update_lam(lam_0, a, b, X):
    N = len(X)
    lam_N = (lam_0 + N) * (a/b)
    return lam_N

def update_a(a_0, N):
    a_N = a_0 + ((N + 1)/2)
    return a_N

def update_b(b_0, mu_0, mu_N, lam_0, lam_N, X):
    xsum = 0
    for x in X:
        xsum += x**2 - 2*x*mu_N + mu_N**2 + 1/lam_N
    
    b_N = b_0 + xsum/2 + 1/2*(lam_0)*(1/lam_N + mu_N 
                            - 2*mu_0*mu_N + mu_0**2) 
    return b_N

def q_mu(x, mu, lam):
    return norm.pdf(x, mu, np.sqrt(1/lam))

def q_tau(y, a, b):
    return gamma.pdf(y, a = a, scale = 1/b)

def qq(x, y, params):
    q = q_mu(x, params.mu, params.lam)*q_tau(y, params.a, params.b)
    return q

def gamma_normal(x, tau, params):
    mu = params.mu #true_mu 
    lam = params.lam#true_tau #params.lam
    beta = params.b
    alfa = params.a
    
    f = ((beta**alfa)*np.sqrt(lam))
    f /= (gamma_f(alfa)*np.sqrt(2*pi))
    f *= (tau**(alfa-1/2))
    f /= (np.exp(beta*tau))
    f /= (np.exp((lam*tau*(x-mu)**2)/2))
    return f

class Params:
    # Just to make function calling cleaner
    def __init__(self, mu, lam, a, b):
        self.mu = mu
        self.lam = lam
        self.a = a
        self.b = b

    def __repr__(self):
        s = str(self.mu)+" "+str(self.lam)+" "+str(self.a)+" "+str(self.b)
        return s

def plot_dists(VI_params, post_params):
    res = 100
    viMap = np.zeros([res, res])
    postMap = np.zeros([res, res])

    mus = np.linspace(true_mu+2, true_mu-2, res)
    taus = np.linspace(true_tau*2, 0.00001, res)

    for j in range(res):
        for i in range(res):
            viMap[j, i] = qq(mus[i], taus[j], VI_params)
            postMap[j, i] = gamma_normal(mus[i], taus[j], post_params)

    plt.contour(mus, taus, viMap, colors = 'red')
    plt.contour(mus, taus, postMap)
    plt.axis([-1.7, 1.7, -0.2, 0.2])
    plt.show()
    
# Generate data
true_mu = 0
true_tau = 0.1
N = 20 # With more datapoints, the true posterior and the VI posterior will be more similar
data = np.random.normal(true_mu, (1/true_tau)**0.5, N)
xsum = 0
xmean = (np.sum(data))/N
for d in data:
    xsum += (d-xmean)**2

# Prior
lam_0 = 0
mu_0 = 0
a_0 = 0
b_0 = 0
a_N = 0.0000000001
b_N = 0.0000000001
lam_N = 0

post_mu = (lam_0*mu_0 + np.sum(data))/(lam_0 + N)
post_lam = lam_0 + N
post_a = a_0 + N/2
post_b = b_0 + 1/2*xsum + (lam_0*N*((xmean - mu_0)**2))/(2*(lam_0 + N)) 

post_params = Params(post_mu, post_lam, post_a, post_b)
err = 1

while err > 0.0000001:
    mu_N = update_mu(mu_0, lam_0, data)
    lam_old = lam_N
    lam_N = update_lam(lam_0, a_N, b_N, data)
    err = (lam_old - lam_N)**2

    a_N = update_a(a_0, N)
    b_N = update_b(b_0, mu_0, mu_N, lam_0, lam_N, data)

VI_params = Params(mu_N, lam_N, a_N, b_N)

print("VI mu and tau:", mu_N, a_N/b_N)
print("Posterior mu and tau:", post_mu, post_a/post_b)
print("True mu and tau:", true_mu, true_tau)
plot_dists(VI_params, post_params)




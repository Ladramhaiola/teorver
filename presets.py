from statistics import mean, pstdev, pvariance
import scipy.stats as stats
from math import erf, sqrt, isclose
from numpy import arange, linalg

a = {
    4: 2.059,
    5: 2.32,
    6: 2.55,
    7: 2.7,
    8: 2.8,
    9: 2.95,
    10: 3.1
}

def me(seq): return mean(seq)

def dispersion(seq, me=None): return pvariance(seq, me)

def sigma(seq=None, me=None):
    if len(seq) > 10: return pstdev(seq, me)
    return (max(seq) - min(seq))/a[len(seq)]

def cov(seq1, seq2): # covariation
    mesq1, mesq2 = me(seq1), me(seq2)
    n = len(seq1)
    return sum([ (x - mesq1)*(y - mesq2) for x, y in zip(seq1, seq2)])/n

def ro(seq1, seq2): # correlation coefficient
    return cov(seq1, seq2)/(sigma(seq1)*sigma(seq2))

def ro_n(seq1, seq2, n=2): # non linear regression, n -> degree
    seq1 = [el**n for el in seq1]
    return cov(seq1, seq2)/(sigma(seq1)*sigma(seq2))

def phi(x): # laplace integral
    return erf(x/sqrt(2))/2.0

def xbyf(phir): # find x from phi(x) = ...
    for x in arange(0, 5, 0.001):
        if isclose(phi(x), phir, rel_tol=1e-03):
            return x

def confprob(interval, n, sigma): # confidence probability
    return 2*phi((interval*sqrt(n))/sigma)

def linearRegressionF(seqx, seqy, x=None): # finds everyting needed for formula
    beta = ro(seqx, seqy)*sigma(seqy)/sigma(seqx)
    alpha = me(seqy) - beta*me(seqx)
    mistake = (1 - ro(seqx, seqy)**2)*sigma(seqy)**2
    if x is None: return "Y = {} + {}X".format(alpha, beta), mistake
    return alpha + beta*x, "Y = {} + {}X".format(alpha, beta) , mistake

def multiregression(seqy, xs, seqxs):
    L, detLjs = [], []
    for i in range(len(seqxs)):
        L.append([cov(seqxs[j], seqxs[i]) for j in range(len(seqxs))])
    detL = linalg.det(L)
    for j in range(len(seqxs)):
        Ljs = L[:]
        for i in range(len(Ljs)):
            Ljs[j][i] = cov(seqxs[j], seqy)
        detLjs.append(linalg.det(Ljs))
    betas = [detLj/detL for detLj in detLjs]
    alpha = me(seqy) - sum([beta*me(seqxs[i]) for beta, i in zip(betas, range(len(seqxs)))])
    return alpha, betas

def chiSquaredS(mseq, n, pseq): # chi^2 for single sequence
    return sum([((mi - n*pi)**2)/(n*pi) for mi, pi in zip(mseq, pseq)])

def chiprobability(chi, l): # chi probability, auto l - 1
    return 1 - stats.chi2.cdf(x=chi, df=l - 1)

d1 = [172, 183, 181, 185, 170, 173, 180, 190, 157, 176] # experimental data
d2 = [40,   42,  42,  43,  40,  41,  42,  44,  35,  41]

#print("me1: ", me(d1))
#print("disp2: ", dispersion(d1))
#print("sigma2: ", sigma(d1))
#print("cov: ", cov(d1, d2))
#print("ro: ", ro(d1, d2))
#print("ro_n: ", ro_n(d1, d2))
#print("phi: ", phi(0.09))
#ch = chiSquaredS([2, 2, 2], 10, [0.3, 0.5, 0.2])
#print(ch)
#print(xbyf(0.2389))
#print(chiprobability(ch, 3))
#print(confprob(2, 42, 7.79))
#print("lr: ", linearRegressionF(d1, d2, 198))
#print(chiprobability(11.07, 6))
#print(linearRegressionF(d1, d2)[0])
#multiregression([1, 1, 4, 5], [2, 3], [[2, 4, 5, 6], [4, 3, 7, 8]])
n = 3
interval = 0.1*0.5*n
sigma = sqrt(0.5*0.5*n)
print(confprob(interval, n, sigma))
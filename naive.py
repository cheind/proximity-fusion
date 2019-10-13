import numpy as np
from scipy.stats import beta
from scipy.stats import multivariate_normal

class Beta:
    '''Beta distribution'''

    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    @staticmethod
    def fit(X):
        a,b = beta.fit(X)[:2]       
        return Beta(a, b)

    def pdf(self, x):
        return beta.pdf(x, self.a, self.b)
    
    def __repr__(self):
        return f'Beta({self.a:.2f},{self.b:.2f})'

class Bernoulli:
    '''Bernoulli distribution'''
    def __init__(self, theta):
        self.theta = theta
        
    @staticmethod
    def fit(X):
        theta =  np.count_nonzero(X) / X.shape[0]
        return Bernoulli(theta)
    
    def pdf(self, x):
        return self.theta**x*(1-self.theta)**(1-x)
    
    def __repr__(self):
        return f'Bernoulli({self.theta:.2f})'
    
class Normal:
    '''Normal distribution'''

    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
        
    @staticmethod
    def fit(X):
        mu, cov = np.mean(X, axis=0), np.cov(X, rowvar=0)
        return Normal(mu,cov) 
    
    def pdf(self, x):
        return multivariate_normal.pdf(x, mean=self.mu, cov=self.cov)
    
    def __repr__(self):
        return f'Normal({self.mu:.2f},{np.sqrt(self.cov):.2f})'
    
class Histogram:
    '''2D histogram'''
    def __init__(self, H=None, data=None, num_bins=3, xyrange=[[-5,5], [-1,3]]):
        if data is None:
            data = np.empty((0,2))            
            
        if H is None:
            self.H, self.xe, self.ye = np.histogram2d(data[:, 0], data[:, 1], bins=num_bins, range=xyrange, normed=False)
            self.H += np.ones_like(self.H)
            self.H /= self.H.sum()
            self.H = self.H.astype(np.float32)
        else:
            K, self.xe, self.ye = np.histogram2d(data[:, 0], data[:, 1], bins=num_bins, range=xyrange, normed=False)
            self.H = np.asarray(H).reshape(K.shape[0], K.shape[1])
        
    def bin_coords(self, xy):
        x = np.atleast_2d(xy)
        bx, by = np.searchsorted(self.xe, x[:, 0]) - 1, np.searchsorted(self.ye, x[:, 1]) - 1 # assume x in range!
        bx = np.clip(bx, 0, self.H.shape[0]-1)
        by = np.clip(by, 0, self.H.shape[1]-1)
        return bx, by
        
    def bin_coords1d(self, xy):
        bx, by = self.bin_coords(xy)
        return bx * self.H.shape[1] + by
    
    
class Categorical2d:
    '''Categorical distribution in xy-plane.'''
    def __init__(self, hist):
        self.hist = hist
        
    @staticmethod
    def fit(X, num_bins, xyrange):   
        h = Histogram(data=X, num_bins=num_bins, xyrange=xyrange)        
        return Categorical2d(h)
    
    def pdf(self, x):
        bx, by = self.hist.bin_coords(x)
        return self.hist.H[bx, by]
    
    def __repr__(self):
        return f'Categorical2d({np.round(self.hist.H*100).astype(int)})'
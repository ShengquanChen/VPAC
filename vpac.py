# -*- coding: utf-8 -*-
import time
import numpy as np
from numpy import transpose as tr
from numpy.linalg import inv
from scipy.linalg import pinvh
from scipy.misc import logsumexp
from scipy.special import psi, gammaln
from sklearn.utils import check_array
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class VPAC(object):
    '''    
    Parameters:
    -----------     
    y : array-like, shape = [n_features, n_samples]
       Observed data

    latent_dim : int, optional (DEFAULT = 5)
       Number of latent dimensions

    n_components : int, optional (DEFAULT = 5)
       Maximum number of mixture components
       
    tol : float, optional (DEFAULT = 1e-5)
       Convergence threshold
       
    n_iter : int, optional (DEFAULT = 100)
       Maximum number of iterations
       
    n_mfa_iter: int, optional (DEFAULT = 1)
       Maximum number of iterations for Mean Field Approximation of lower bound
       
    n_init: int , optional (DEFAULT = 50)
       Number of restarts in initialization
       
    prune_thresh: float, optional (DEFAULT = 0)
       Threshold for cluster removal. If weight corresponding to cluster becomes smaller than threshold it is removed.
     
    init_params: dict, optional (DEFAULT = {})
       Initial parameters for model (keys = ['dof','covar','weights','gamma','means'])
           'dof'    : int  
                  Degrees of freedom for prior distribution
           'covar'  : array of size (n_features, n_features)
                  Inverse of scaling matrix for prior wishart distribution
           'weights': array of size (n_components,) 
                  Latent variable distribution parameter (cluster weights)
           'gamma'  : float
                  Scaling constant for precision of mean's prior 
           'means'  : array of size (n_components, n_features) 
                  Means of clusters

    verbose: bool, optional (DEFAULT = False)
       Enables verbose output (require the 'tqdm' package)

    '''

    def __init__(self, y, latent_dim = 5, n_components = 5, log_trans = True, tol = 1e-5, n_iter = 100, n_mfa_iter = 1,
                 n_init = 50, prune_thresh = 0, init_params = dict(), verbose = False, hyper=None):
        if log_trans:
            y = np.log(1+y)
        self.y = y
        self.p = y.shape[0]
        self.q = latent_dim
        self.n = y.shape[1]
        if hyper is None:
            self.hyper = HyperParameters()
        else:
            self.hyper = hyper
        self.q_dist = Qdistribution(self.n, self.p, self.q)

        self.n_mfa_iter    = n_mfa_iter
        self.prune_thresh  = prune_thresh
        self.n_iter        = n_iter
        self.n_init        = n_init
        self.n_components  = n_components
        self.tol           = tol
        self.init_params   = init_params
        self.verbose       = verbose
        if self.verbose:
            from tqdm import tqdm_notebook
               

    def _init_params(self):
        time_start = time.time()
        pca = PCA(n_components = self.q, random_state = 0)
        self.X0 = pca.fit_transform(self.y.T)

        if 'means' in self.init_params:
            means0   = self.init_params['means']
        else:
            kms = KMeans(n_init = self.n_init, n_clusters = self.n_components)
            means0 = kms.fit(self.X0).cluster_centers_
        if 'covar' in self.init_params:
            scale_inv0 = self.init_params['covar']
            scale0     = pinvh(scale_inv0)
        else:
            diag_els   = np.abs(np.max(self.X0,0) - np.min(self.X0,0))/2
            scale_inv0 = np.diag(diag_els)
            scale0     = np.diag(1. / diag_els)
        if 'weights' in self.init_params:
            weights0   = np.ones(self.n_components) / self.n_components
        else:
            weights0   = np.ones(self.n_components) / self.n_components
        if 'dof' in self.init_params:
            dof0 = self.init_params['dof']
        else:
            dof0 = self.q
        if 'gamma' in self.init_params:
            gamma0 = self.init_params['gamma']
        else:
            gamma0 = 1e-3
        self.active  = np.ones(self.n_components, dtype = np.bool)
        
        assert dof0 >= self.q,('Degrees of freedom should be larger than dimensionality of data')
        assert means0.shape[0]   == self.n_components,('Number of centrods defined should be equal to number of components')
        assert means0.shape[1]   == self.q,('Dimensioanlity of means and data should be the same')
        assert weights0.shape[0] == self.n_components,('Number of weights should be equal to number of components')

        scale   = np.array([np.copy(scale0) for _ in range(self.n_components)])
        means   = np.copy(means0)
        weights = np.copy(weights0)
        dof     = dof0 * np.ones(self.n_components)
        gamma   = gamma0 * np.ones(self.n_components)
        init_   = [means0, scale0, scale_inv0, gamma0, dof0, weights0]
        iter_   = [means, scale, scale_inv0, gamma, dof, weights]

        time_end = time.time()
        if self.verbose:
            print(('Initial time is %0.3f seconds.' % (time_end - time_start)))
        return init_, iter_
        
        
    def fit(self):
        n_samples    = self.y.shape[1]
        init_, iter_ = self._init_params()
        means0, scale0, scale_inv0, gamma0, dof0, weights0 = init_
        means, scale, scale_inv, gamma, dof, weights       = iter_
        active = np.ones(self.n_components, dtype = np.bool)
        self.n_active = np.sum(active)
        X = self.X0
        z_update_flag = 0
        time_start = time.time()
        if self.verbose:
            print('%d training samples with %d features.' % (n_samples, self.p))
            pbar = tqdm_notebook(total=self.n_iter)

        z_record = np.zeros((self.q, self.n))

        for j in range(self.n_iter):
            self.update_mu()
            self.update_w()
            self.update_alpha()
            self.update_tau()

            means_before = np.copy(means)
            for i in range(self.n_mfa_iter):
                resps, delta_ll = self._update_resps_parametric(X, weights, self.n_active, dof, means, scale, gamma)
                Nk = np.sum(resps,axis = 0)
                Xk = [np.sum(resps[:,k:k+1]*X,0) for k in range(self.n_active)]
                Sk = [np.dot(resps[:,k]*X.T,X) for k in range(self.n_active)]
                gamma, means, dof, scale = self._update_params(Nk, Xk, Sk, gamma0, means0, dof0, scale_inv0, gamma, means, dof, scale)

            self.update_z(s = resps, T = (scale, dof), m = means)
            if z_update_flag:
                X = self.transform().T
            
            weights        = Nk + weights
            weights       /= np.sum(weights)
            active         = weights >= self.prune_thresh
            means0         = means0[active,:]
            scale          = scale[active,:,:]
            weights        = weights[active]
            weights       /= np.sum(weights)
            dof            = dof[active]
            gamma          = gamma[active]
            n_comps_before = self.n_active
            means          = means[active,:]
            self.n_active  = np.sum(active)

            if self.verbose:
                pbar.update(1)

            if n_comps_before == self.n_active:
                if self._check_convergence(n_comps_before, means_before, means):
                    if z_update_flag > 3:
                        if self.verbose:
                            print("Algorithm converged")
                        break
                    z_update_flag += 1

        time_end = time.time()
        if self.verbose:
            print(('Fitting time is %0.3f seconds.' % (time_end - time_start)))

        self.means_      = means
        self.weights_    = weights
        self.covars_     = np.asarray([1./df * pinvh(sc) for sc,df in zip(scale,dof)])
        self.predictors_ = self._predict_dist_params(dof,gamma,means,scale)
        return z_record
        

    def update_z(self, s, T, m):
        E_s = s
        self.E_s = E_s
        scale, dof = T
        scale_inv = np.asarray([pinvh(sc) for sc in scale])
        E_m = m
        E_T = np.asarray([df * sc for sc,df in zip(scale_inv,dof)])

        E_s_E_T = np.tensordot(E_s, E_T, axes=([1],[0]))
        E_m_E_T = np.asarray([tr(em).dot(et) for em,et in zip(E_m,E_T)])
        self.E_m_E_T = E_m_E_T

        q = self.q_dist
        tau_mean = q.tau_a / q.tau_b
        q.z_cov  = inv(tau_mean * tr(q.w_mean).dot(q.w_mean) + E_s_E_T)
        tr_w_mean = tr(q.w_mean)

        term1 = tau_mean * np.asarray([(q.z_cov[i].dot(tr_w_mean)).dot(self.y[:,i] - q.mu_mean) for i in range(self.n)])
        term2 = np.asarray([q.z_cov[i].dot(tr(E_s[i].dot(E_m_E_T))) for i in range(self.n)])
        q.z_mean = tr(term1 + term2)
        
    def update_w(self):
        q = self.q_dist
        z_cov = np.zeros((self.q, self.q))
        for n in range(self.n):
            x      = q.z_mean[:, n]
            z_cov += x[:, np.newaxis].dot(np.array([x]))
        q.w_cov = np.diag(q.alpha_a / q.alpha_b) + q.tau_mean() * z_cov
        q.w_cov = inv(q.w_cov)
        yc = self.y - q.mu_mean[:, np.newaxis]
        q.w_mean = q.tau_mean() * q.w_cov.dot(q.z_mean.dot(tr(yc)))
        q.w_mean = tr(q.w_mean)

    def update_mu(self):
        q = self.q_dist
        tau_mean = q.tau_a / q.tau_b
        q.mu_cov   = (self.hyper.beta + self.n * tau_mean)**-1 * np.eye(self.p)
        q.mu_mean  = np.sum(self.y - q.w_mean.dot(q.z_mean), 1)
        q.mu_mean  = tau_mean * q.mu_cov.dot(q.mu_mean)

    def update_alpha(self):
        q = self.q_dist
        q.alpha_a = self.hyper.alpha_a + 0.5 * self.p
        q.alpha_b = self.hyper.alpha_b + 0.5 * np.linalg.norm(q.w_mean, axis=0)**2

    def update_tau(self):
        q = self.q_dist
        q.tau_a = self.hyper.tau_a + 0.5 * self.n * self.p
        q.tau_b = self.hyper.tau_b
        w  = q.w_mean
        ww = tr(w).dot(w)
        for n in range(self.n):
            y = self.y[:, n]
            x = q.z_mean[:, n]
            q.tau_b += (y.dot(y) + q.mu_mean.dot(q.mu_mean)) / 2
            q.tau_b += (np.trace(ww.dot(x[:, np.newaxis].dot([x])))) / 2
            q.tau_b += (2.0 * q.mu_mean.dot(w).dot(x[:, np.newaxis])) / 2
            q.tau_b -= (2.0 * y.dot(w).dot(x)) / 2
            q.tau_b -= (2.0 * y.dot(q.mu_mean)) / 2
        
    def _update_params(self, Nk, Xk, Sk, gamma0, means0, dof0, scale_inv0, gamma, means, dof, scale):
        for k in range(self.n_active):
            gamma[k] = gamma0 + Nk[k]
            means[k] = (gamma0*means0[k,:] + Xk[k]) / gamma[k]
            dof[k]   = dof0 + Nk[k] + 1
            scale[k,:,:]  = pinvh( scale_inv0 + (gamma0*Sk[k] + Nk[k]*Sk[k] - 
                                 np.outer(Xk[k],Xk[k]) - 
                                 gamma0*np.outer(means0[k,:] - Xk[k],means0[k,:])) /
                                 (gamma0 + Nk[k]) )
        return gamma,means,dof,scale


    def _update_logresp_cluster(self, X, k, weights, dof, means, scale, gamma):
        d = X.shape[1]
        scale_logdet   = np.linalg.slogdet(scale[k] + np.finfo(np.double).eps)[1]
        e_logdet_prec  = sum([psi(0.5*(dof[k]+1-i)) for i in range(1,d+1)])
        e_logdet_prec += scale_logdet + d*np.log(2)
        x_diff         = X - means[k,:]
        e_quad_form    = np.sum( np.dot(x_diff,scale[k,:,:])*x_diff, axis = 1 )
        e_quad_form   *= dof[k]
        e_quad_form   += d / gamma[k] 
        
        log_pnk        = np.log(weights[k] / np.sum(weights)) + 0.5*e_logdet_prec - 0.5*e_quad_form
        log_pnk       -= d * np.log( 2 * np.pi)
        return log_pnk


    def _update_resps_parametric(self, X, log_weights, clusters, *args):
        log_resps  = np.asarray([self._update_logresp_cluster(X,k,log_weights,*args) for k in range(clusters)]).T
        log_like       = np.copy(log_resps)
        log_resps     -= logsumexp(log_resps, axis = 1, keepdims = True)
        resps          = np.exp(log_resps)
        delta_log_like = np.sum(resps*log_like) - np.sum(resps*log_resps)
        return resps, delta_log_like
                                                 
                             
    def _check_convergence(self, n_components_before, means_before, means):
        conv = True
        for mean_before,mean_after in zip(means_before,means):
            mean_diff = mean_before - mean_after
            conv  = conv and np.sum(np.abs(mean_diff)) / means.shape[1] < self.tol
        return conv
        
        
    def _predict_dist_params(self, dof, gamma, means, scale):
        d = means.shape[1]
        predictors = []
        for k in range(self.n_active):
            df     = dof[k] + 1 - d
            prec   = scale[k,:,:] * gamma[k] * df / (1 + gamma[k])
            predictors.append(StudentMultivariate(means[k,:],prec,dof[k],d))
        return predictors

    def transform(self, y=None):
        if y is None:
            return self.q_dist.z_mean
        q = self.q_dist
        tau_mean = q.tau_a / q.tau_b
        term1  = tau_mean * np.asarray([(q.z_cov[i].dot(tr(q.w_mean))).dot(y[:,i] - q.mu_mean) for i in range(y.shape[1])])
        term2  = np.asarray([q.z_cov[i].dot(tr(self.E_s[i].dot(self.E_m_E_T))) for i in range(y.shape[1])])
        z_mean = tr(term1 + term2)
        return z_mean
        
    def predict_proba(self, y=None):
        if y is None:
            y_transform = self.q_dist.z_mean
        else:
            y_transform = self.transform(y)
        X = y_transform.T
        X       = check_array(X)
        pr      = [st.logpdf(X) + np.log(lw) for st,lw in zip(self.predictors_,self.weights_)]
        log_probs   = np.asarray(pr).T 
        log_probs  -= logsumexp(log_probs, axis = 1, keepdims = True)
        return np.exp(log_probs)



class HyperParameters(object):
    def __init__(self):
        self.alpha_a = 0.001
        self.alpha_b = 0.001
        self.tau_a   = 0.001
        self.tau_b   = 0.001
        self.beta    = 0.001

class Qdistribution(object):
    def __init__(self, n, p, q):
        self.n = n
        self.p = p
        self.q = q
        self.init_rnd()

    def init_rnd(self):
        self.z_mean  = np.random.normal(0.0, 1.0, self.q * self.n).reshape(self.q, self.n)
        self.z_cov   = np.eye(self.q)
        self.w_mean  = np.random.normal(0.0, 1.0, self.p * self.q).reshape(self.p, self.q)
        self.w_cov   = np.eye(self.q)
        self.alpha_a = 1.0
        self.alpha_b = np.empty(self.q)
        self.alpha_b.fill(1.0)
        self.mu_mean = np.random.normal(0.0, 1.0, self.p)
        self.mu_cov  = np.eye(self.p)
        self.tau_a   = 1.0
        self.tau_b   = 1.0

    def tau_mean(self):
        return self.tau_a / self.tau_b

    def alpha_mean(self):
        return self.alpha_a / self.alpha_b

class StudentMultivariate(object):
    def __init__(self,mean,precision,df,d):
        self.mu   = mean      
        self.L    = precision 
        self.df   = df       
        self.d    = d        
                
    def logpdf(self,x):
        xdiff     = x - self.mu
        quad_form = np.sum(np.dot(xdiff,self.L)*xdiff, axis = 1)
        return ( gammaln( 0.5 * (self.df + self.d)) - gammaln( 0.5 * self.df ) +
                 0.5 * np.linalg.slogdet(self.L)[1] - 0.5*self.d*np.log(self.df*np.pi) -
                 0.5 * (self.df + self.d) * np.log(1 + quad_form / self.df) )
        
    def pdf(self,x):
        return np.exp(self.logpdf(x))
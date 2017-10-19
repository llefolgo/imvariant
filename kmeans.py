import numpy as np
import time
from fortran import interface as f90

class Kmeans():
    def __init__(self,X,K,verbose=False,paranoid=False,round_input=True):
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.K = K
        self.r = np.zeros((self.N,self.K),dtype=bool)
        self.mu = np.zeros((self.K,self.D),dtype=np.float32)
        self.X = np.asarray(X,dtype=np.float32)
 
        # criterion for convergence in optimization of means
        self.convergence_criterion = None
        self.convergence_tol = None
        self.convergence_buffer = []
        
        # verbosity
        self.verbose = verbose
        self.paranoid = paranoid
        
        # performance
        self.round = round_input
        self.Nhist = None

        if self.round: self.round_input()

    def round_input(self):
        # decimals to round to
        #precision = 6
            
        t1 = time.time()
        #if self.verbose: print ('rounding input to {} decimals of precision'.format(precision))
        
        if self.D==1:
            self.Nhist,edges = np.histogram(self.X,bins=1000)
            self.X = np.asarray([0.5*(edges[ii]+edges[ii+1]) for ii in range(1000)],dtype=np.float64)

            # remove zero elements
            idx = np.nonzero(self.Nhist)[0]

            self.Nhist = self.Nhist[idx]
            self.X = self.X[idx]

            # resize arrays
            self.N = self.X.shape[0]
            self.r = np.zeros((self.N,self.K),dtype=bool)

            ## reformat so 2nd dimension is available to query
            self.X = np.reshape(self.X,(self.N,self.D))
            
            if self.verbose: print ('finished rounding input')
        else:
            raise NotImplementedError
        t2 = time.time()
        print ('rounding took {}s'.format(t2-t1))
    def optimization_converged(self):
        """
        criterion for convergence
        --------------------------
        1. objective_measure - converge the objective measure to tol per observation
        2. mean - converge gaussian means to tol for the largest varying mean
        """
        
        if self.convergence_criterion not in ["objective_measure","mean"]:
            KmeansUserError("{} is not a supported convergence criterion".format(self.convergence_criterion))
            
        if self.convergence_criterion == "objective_measure":
            # want convergence measure to be independent of number of observations
            current = self.get_objective_measure()/float(self.N)
            
            self.convergence_buffer.append(current)
            
            if len(self.convergence_buffer)<3:
                # at very beginning of optimization
                result = False
            else:     
                if all([abs(self.convergence_buffer[ii]-self.convergence_buffer[ii-1])<self.convergence_tol \
                         for ii in range(1,3)]):
                    # less susceptible to local plateaus when considering convergence over 2 iterations
                    result = True
                else:
                    self.convergence_buffer.pop(0)
                    result = False
        
        elif self.convergence_criterion == "mean":
            self.convergence_buffer.append(self.mu)
            
            if len(self.convergence_buffer)<3:
                result = False
            else:
                if all([max(np.linalg.norm(self.convergence_buffer[ii]-self.convergence_buffer[ii-1],axis=1)) < \
                        self.convergence_tol for ii in range(1,3)]):
                    result = True
                else:
                    self.convergence_buffer.pop(0)
                    result = False
        if self.verbose: print ('evaulating convergence for : {} result = {}'.format(self.convergence_buffer,result))
        
        return result
        
    def initialise(self,seed=None):
        np.random.seed(seed=seed)

        # initialise with K random means selected from input data
        self.mu = self.X[np.random.choice(range(self.N),self.K,replace=False)]
    def optimize(self,seed=None,criterion="objective_measure",tol=1e-3):
        self.initialise(seed=seed)
        
        self.convergence_criterion = criterion.lower()
        self.convergence_tol = tol
        
        for ii in range(1000):
            self.update_assignment()
            self.update_mean()
            if self.optimization_converged():
                # have met criterion
                break
    def update_assignment(self):
        #self.r[:,:] = 0
        #for ii in range(self.N):
        #    self.r[ii,np.argmin([np.linalg.norm(self.X[ii,:]-self.mu[kk,:]) for kk in range(self.K)])] = 1

        self.r = f90.calculate_softassignment(x=self.X,mean=self.mu)

        if self.paranoid:
            tmpr = np.zeros((self.N,self.K),dtype=np.float32)

            for ii in range(self.N):
                for kk in range(self.K):
                    # distance^2 between xi and muk
                    for dd in range(self.D):
                        tmpr[ii,kk] += (self.X[ii,dd]-self.mu[kk,dd])**2 
                # min kk
                idx = np.argmin(tmpr[ii,:])

                for kk in range(self.K):
                    if kk!=idx:
                        tmpr[ii,kk] = 0
                    else:
                        tmpr[ii,kk] = 1
            # raise error if not equal
            np.testing.assert_array_almost_equal(self.r,tmpr)

    def update_mean(self):
        #for kk in range(self.K):
        #    # meed to cope with when no points are assigned to class kk
        #    if np.sum(self.r.T[kk,:])>0:
        #    #if len ([1 for ii in range(self.N) if self.r[ii,kk]])>0:
        #        self.mu[kk] = np.sum([self.X[ii,:]*self.Nhist[ii] for ii in range(self.N) if self.r[ii,kk]])/\
        #                np.inner(self.r.T[kk,:],self.Nhist)
        #    else:
        #        # do not change class location if it is empty
        #        pass
        #self.mu = np.asarray(self.mu)
    
        self.mu = f90.calculate_mean(x=self.X,r=self.r,Nhist=self.Nhist)

        if self.paranoid:
            tmpmu = np.zeros((self.K,self.D),dtype=np.float64)
            for kk in range(self.K):
                cntr = 0
                for ii in range(self.N):
                    if self.r[ii,kk] == 1:
                        cntr += self.Nhist[ii]
                        for dd in range(self.D):
                            tmpmu[kk,dd] += self.X[ii,dd]*self.Nhist[ii]
                if cntr>0:
                    for dd in range(self.D):
                        tmpmu[kk,dd] /= float(cntr)
            # raise error if not equal 
            np.testing.assert_array_almost_equal(self.mu,tmpmu)
        
        self.checks("update_mean")
        
    def get_objective_measure(self):
        #measure = np.sum([np.sum([np.linalg.norm(self.X[ii,:]-self.mu[kk,:])**2*self.Nhist[ii] for kk in range(self.K) if self.r[ii,kk]]) \
        #        for ii in range(self.N)])
    
        measure = f90.calculate_objective_measure(x=self.X,mean=self.mu,Nhist=self.Nhist,r=self.r)

        if self.paranoid:
            tmpmeasure = 0.0
            for ii in range(self.N):
                for kk in range(self.K):
                    if self.r[ii,kk] == 1:
                        for dd in range(self.D):
                            tmpmeasure += (self.X[ii,dd]-self.mu[kk,dd])**2*self.Nhist[ii]
            # could have some FP arithmetic error here
            np.testing.assert_almost_equal(measure,tmpmeasure)
        return measure

    def get_means(self):
        """
        return current cluster means
        """
        return self.mu
    
    def get_variances(self):
        """
        return empirical variance, var_k = sum_{n=1}^N r_{nk} | x_n - mu_k |^2     / sum_{n=1}^N r_{nk}
        """
        var = np.zeros(self.K,dtype=np.float64)
        for kk in range(self.K):
            if len([True for ii in range(self.N) if self.r[ii,kk]])>0:
                var[kk] = np.sum([np.linalg.norm(self.X[ii,:]-self.mu[kk,:])**2*self.Nhist[ii] for ii in range(self.N) if self.r[ii,kk]])/\
                        np.inner(self.r.T[kk,:],self.Nhist)
        
        if self.paranoid:
            tmpvar = np.zeros(self.K,dtype=np.float64)
            cntrs = np.zeros(self.K,dtype=int)
            for ii in range(self.N):
                for kk in range(self.K):
                    if self.r[ii,kk]==1:
                        cntrs[kk] += self.Nhist[ii]
                        tmp=0.0
                        for dd in range(self.D):
                            tmp += (self.X[ii,dd]-self.mu[kk,dd])**2 *self.Nhist[ii]
                        tmpvar[kk] += tmp
            for kk in range(self.K):
                tmpvar[kk] /= float(cntrs[kk])

            np.testing.assert_array_almost_equal(var,tmpvar)
        return var
    
    def get_occupancies(self):
        """
        return occupancies_k = sum_n rnk
        """
        occupancies =  np.inner(self.r.T,self.Nhist)

        if self.paranoid:
            tmpoccupancies = np.zeros(self.K,dtype=np.float64)

            for ii in range(self.N):
                for kk in range(self.K):
                    if self.r[ii,kk]==1:
                        tmpoccupancies[kk] += self.Nhist[ii]
            np.testing.assert_array_almost_equal(occupancies,tmpoccupancies)
        
        return occupancies

    def get_N(self):
        return self.N

    def clear_convergence_buffer(self):
        self.convergence_buffer = []

    def checks(self,method):
        if method == "update_mean":
            if any(np.isinf(self.mu).flatten()):
                raise KmeansSevereError("some mean values are np.inf {}".format(self.mu))
            elif any(np.isnan(self.mu).flatten()):
                raise KmeansSevereError("some mean values are np.nan {}".format(self.mu))

class sample_kmeans(Kmeans):
    def __init__(self,X,K,Nsamples=10,verbose=False,paranoid=False,round_input=True):
        """
        class to sample a number Nsamples of initial means 
        """
        # generate rounded data and histograms
        super().__init__(X,K,verbose,paranoid,round_input)
        
        # data for collected samples over initial means
        self.Nsamples = Nsamples
        self.means = []
        self.variances = []
        self.occupancies = []
        self.lower_bounds = []

    def sample(self):
        """
        run samples, storing mean,variance and lower bound each time
        """
        for nn in range(self.Nsamples):
            # seed with ms since method call
            super().optimize(seed=int(time.clock()*1e3))

            #--------------#
            # book keeping #
            #--------------#
            self.means.append(super().get_means())
            self.variances.append(super().get_variances())
            self.occupancies.append(super().get_occupancies())
            
            # the lower bound per data point is a more meaningful convergence measure
            self.lower_bounds.append( super().get_objective_measure()/float(super().get_N()) )
            
            # clear lower bounds stored to evaluate convergence
            super().clear_convergence_buffer()
    
    def get_means(self):
        """
        return mixture means from the best sample
        """
        return self.means[np.argmin(self.lower_bounds)]

    def get_variances(self):
        """
        return mixture empirical variances from the best sample
        """
        return self.variances[np.argmin(self.lower_bounds)]

    def get_occupancies(self):
        """
        return mixture occupancies from the best sample
        """
        return self.occupancies[np.argmin(self.lower_bounds)]

class KmeansUserError(Exception):
    pass    
class KmeansSevereError(Exception):
    pass    

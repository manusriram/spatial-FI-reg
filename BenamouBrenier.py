import numpy as np
import matplotlib.pyplot as plt
import warnings
from time import time
from warnings import warn
from matplotlib.widgets import Slider
from poisson import laplacian_matrix,poisson,derivative_matrices,divergence,gradient,poisson2,build_Asolve

class TransportProblem:
    def __init__(self,mesh,mu,nu,T,tau=1,display=True,Afactorized=True):
        (d,*space_grid_shape) = mesh.shape
        if d!=2:
            warn(str(d)+"D space, poisson step not implemented.")
        space_grid_shape = tuple(space_grid_shape)
        if space_grid_shape != mu.shape or space_grid_shape != nu.shape:
            Exception("The space grid dimensions of the mesh doesn't match the mesures")
        spacetime_grid_shape = (T,) + space_grid_shape

        self.spacetime_grid_shape = spacetime_grid_shape
        self.d = d
        self.mesh = mesh 
        self.mu = mu
        self.nu = nu

        self.T = T
        self.times = np.linspace(0,1,T)
        self.rho = (1-self.times.reshape((T,)+d*(1,)))*mu + self.times.reshape((T,)+d*(1,))*nu
        eps = 0.1
        self.rho = (1-eps)*self.rho + eps/np.prod(spacetime_grid_shape)
        self.m = np.zeros((self.d,) + spacetime_grid_shape)
        self.M = np.concatenate((self.rho[np.newaxis,...], self.m))

        self.phi = np.zeros(spacetime_grid_shape)
        self.nabla_phi = np.zeros((d+1,) + spacetime_grid_shape)
        self.laplacian_matrix = laplacian_matrix(space_grid_shape[0])
        self.An, self.Ap = derivative_matrices(space_grid_shape[0])
        self.Afactorized = Afactorized
        if Afactorized == True:
            self.Asolve = build_Asolve(self.T)

        self.a = np.zeros(spacetime_grid_shape)
        self.b = np.zeros((d,)+ spacetime_grid_shape)
        self.c = np.concatenate((self.a[np.newaxis,...], self.b))

        self.tau=tau
        if display:
            print("TransportProblem object initialized.")

    def __str__(self):
        return  str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))
    
    def update_c(self):
         self.c = np.concatenate((self.a[np.newaxis,...], self.b))

    def update_rho_m(self):
        self.rho = self.M[0]
        self.m = self.M[1:]

    def poisson_step1(self,display=False):
        if self.d!=3:
            Exception("Poisson step is implemented only for a 2D space, not for a "
                      + str(self.d) + "D space.")
        g = np.stack((self.mu, self.nu))/self.tau - self.rho[[0,-1]]/self.tau + self.a[[0,-1]]
        field = self.c - self.M/self.tau    
        field_for_div = np.transpose(field, (0, 2, 3, 1)) 
        f = divergence(field_for_div, self.An, self.Ap)

        self.phi = poisson(f,g,self.laplacian_matrix)
        self.nabla_phi = gradient(self.phi,self.An,self.Ap)
        if display:
            print("Poisson step done.")

    def poisson_step2(self,display=False):
        if self.d!=3:
            Exception("Poisson step is implemented only for a 2D space, not for a "
                      + str(self.d) + "D space.")
        g = np.stack((self.mu, self.nu))/self.tau - self.rho[[0,-1]]/self.tau + self.a[[0,-1]]
        field = self.c - self.M/self.tau         
        field_for_div = np.transpose(field, (0, 2, 3, 1)) 
        f = divergence(field_for_div, self.An, self.Ap)
        self.phi = poisson2(f,g,self.Asolve)
        self.nabla_phi = gradient(self.phi,self.An,self.Ap)
        if display:
            print("Poisson step done.") 

    def poisson_step(self,display=False):
        if self.Afactorized==True:
            self.poisson_step2(display=False)
        else:
            self.poisson_step1(display=False)

    def projection_step(self,display=False):
        a=self.a
        b=self.b
        alpha_beta = self.nabla_phi + self.M / self.tau
        tol = 1e-8
        if np.max(alpha_beta) <= 0+tol:
            return
        f = lambda t,alpha,beta: (alpha-0.5*t)*(1+0.5*t)**2 + 0.5*beta**2
        df = lambda t,alpha: (0.5*t+1)*(-0.75*t+alpha-0.5)
        maxiter = 50 
        nbr_maxiter_reached = 0
        iterator = np.ndindex(self.spacetime_grid_shape)
        if display:
            iterator = tqdm(iterator,total=np.prod(self.spacetime_grid_shape))
        for index in iterator: 
            temps = time()
            alpha = alpha_beta[0][index]
            beta = np.linalg.norm(alpha_beta[1:][(...,*index)], axis=0) 

            t = (alpha-0.5)*4/3 + 100 
            i=0
            im_f = f(t,alpha,beta)
            while np.abs(im_f) > tol and i < maxiter:
                t = t - im_f/df(t,alpha)
                im_f = f(t,alpha,beta)
                i=i+1
            if i==maxiter:
                nbr_maxiter_reached = nbr_maxiter_reached + 1
                warn("Max number of iterations reached in Newton's method.")
            
            a[index] = alpha - 1/2*t 
            b[(...,*index)] = alpha_beta[1:][(...,*index)]/(1/2*t+1)
            a[index] = - 1/2*np.sum(alpha_beta[1:][(...,*index)]**2, axis=0)
        self.a=a
        self.b=b
        self.update_c()

        if nbr_maxiter_reached > 0:
            warn("Max number of iterations reached in Newton's method ("+str(nbr_maxiter_reached)+" times).")
        elif display:
            print("Projection step converged to tolerance.")
        return
    
    def projection_step_bis(self, display=False):
        alpha_beta = self.nabla_phi + self.M / self.tau
        alpha,beta = alpha_beta[0], np.sqrt(np.sum(alpha_beta[1:]**2,axis=0))
        iterator = np.ndindex(self.spacetime_grid_shape)
        if display:
            iterator = tqdm(iterator,total=np.prod(self.spacetime_grid_shape))
        for index in iterator: 
            if np.max(alpha[index] + beta[index]**2/2,) > 0:
                a,b,c = 4-2*alpha[index], 4-8*alpha[index], 4*beta[index]**2-8*alpha[index]
                t = last_root(a,b,c)
                self.a[index] = alpha[index] - 1/2*t
                self.b[(...,*index)] = alpha_beta[1:][(...,*index)]/(1/2*t+1)
        self.update_c()
        if display:
            print("Projection step done.")

    def dual_step(self,display=False):
        self.M = self.M - self.tau*(self.c - self.nabla_phi)
        self.update_rho_m()
        if display:
            print("Dual step done.")

    def residual(self):
        return self.nabla_phi[0] + 0.5 * np.sum(self.nabla_phi[1:]**2,axis=0)
    
    def criterium(self):
        try:
            return np.sum(self.rho * np.abs(self.residual())) / np.sum(self.rho * np.sum(self.nabla_phi[1:]**2,axis=0))
        except ZeroDivisionError:
            warn("Division by zero in criterium (rho * nabla_phi = 0), infinity returned.")
            return np.inf
    def lagrangian(self):
        G = np.sum(self.rho[0] * self.phi[0] - self.rho[-1] * self.phi[-1])
        constraint = self.nabla_phi - self.c
        L = G + np.sum(self.M * constraint)
        L_tau = L + self.tau/2 * np.sum(constraint**2)
        return L,L_tau
        
    def solve(self,tol=1e-7,maxiter=100,display=True):
        criteria=[]
        LL=[]
        iterator = range(maxiter)
        if display:
            iterator = tqdm(iterator)
        for i in iterator:
            L,L_tau = self.lagrangian()
            LL.append(L_tau)
            self.poisson_step()
            L,L_tau = self.lagrangian()
            LL.append(L_tau)
            self.projection_step_bis()
            L,L_tau = self.lagrangian()
            LL.append(L_tau)
            self.dual_step()
            self.tau = 2*self.tau
            crit = self.criterium()
            res = np.max(np.abs(self.residual()))
            criteria.append(crit)
            #if res < tol:
                #break
        if res < tol and display:
            print(f"Benamou-Brenier method converged to tolerance with criterium ={crit:.2f}")
        elif display:
           print("Benamou-Brenier method stopped at the maximum number of iterations, with criterium = "+str(crit)+">"+str(tol))
        return criteria,LL
    
    def plot(self,t=None):
        if t is None:
            fig, (ax1,ax2) = plt.subplots(2)
            self.s = Slider(ax = ax2, label = 'value', valmin = 0, valmax = self.T-1, valinit = 1)
            def update(val):
                value=int(self.s.val)
                ax1.cla()
                ax1.contour(self.rho[value])
            self.s.on_changed(update)
            update(0)
            plt.show()
            return fig
        else:
            tt = np.atleast_1d(t)
            for i,t in enumerate(tt):
                plt.figure()
                plt.contourf(np.arange(self.T)/(self.T-1),np.arange(self.T)/(self.T-1),self.rho[int(t*(self.T-1))])
                plt.title("t="+str(t))
                plt.colorbar()
                plt.savefig(f"graphics/rho({i}).pdf")

    def k(rho,m): 
        tol = 1e-7
        norm2_m = np.sum(m**2)
        if rho >= tol:
            return 0.5 * norm2_m / rho
        if np.abs(rho) < tol:
            if norm2_m < tol:
                return 0
            else:
                return np.inf
        if rho <= -tol:
            return np.inf
        
def last_root(a,b,c):
    p = b - a**2/3
    q = a / 27 * (2*a**2 - 9*b) + c
    delta = (p/3)**3 + (q/2)**2
    if delta>0:
        u = np.cbrt(-q/2 + np.sqrt(delta))
        v = np.cbrt(-q/2 - np.sqrt(delta))
        x = u + v - a/3
    elif delta == 0:
        u = np.cbrt(-q/2)
        x = 2*np.abs(u) - a/3
    else:
        u = (-q/2 + 1j*np.sqrt(-delta))**(1/3)
        x = 2*np.real(u) - a/3
    return x

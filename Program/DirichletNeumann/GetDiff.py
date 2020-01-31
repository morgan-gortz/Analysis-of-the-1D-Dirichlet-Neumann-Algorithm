import Algorithm as alg
import Discretisations as disc
import scipy as sc
import numpy as np
import Solver as sol

def Ldense(x):
    return sc.expm1(x)/(sc.e-1)

def Rdense(x):
    return (sc.log(x*(sc.e-1)+1))


def get_largest_div(approxs,exact,conv_rate):
    vals=[]
    for i in range(1,len(approxs)):
        if(abs(approxs[i-1]-exact)<1e-15):
            vals.append(0)
            break
        vals.append(abs(conv_rate-abs((approxs[i]-exact)/(approxs[i-1]-exact))))
    return max(vals)

"""
Arguments:
Takes a dirichlet discretisation, a neumann discretisation,  [x1,gx,x2], u(x1)=b11, u(x2)=b22, lamda1, lambda2, n_dir, n_neu
Returns a list of the largest devation from the convergence rate for [equi,uni,ldense,rdense] grids.

Key arguments:
tests, how many times do we do the same problem (returns average)
mit, number of iterations performed
tl, tolerance when algorithm stops
solver, what solver is used in the discretistions
dirEqui, force equidistant on Dirichlet discretisation
neuEqui, force equidistant on Neumann discretisation
esol, returns an approximation of u(gx) 
"""
def getDiff(discdir,discneu,x1,gx,x2,b11,b22,lmda1,lmda2,n1,n2, tests = 5, mit = 10, tl = 1e-13,solver = sol.SPSolver(),dirEqui = False, neuEqui=False,esol = disc.efemfem.exactFEMFEM,f1 = lambda x: 0.0, f2 = lambda x:0.0):
    x1 = float(x1)
    gx = float(gx)
    x2 = float(x2)
    b11 = float(b11)
    b22 = float(b22)
    lmda1 = float(lmda1)
    lmda2 = float(lmda2)

    conv_rate = (float(lmda1)/float(lmda2))*(float(x2-gx)/float(gx-x1))
    k1 = 0
    for i in range(tests):
        x11 = sc.linspace(x1,gx,n1+2)
        x12 = sc.linspace(gx,x2,n2+2)
        dirich = discdir(x11,solver)
        neum = discneu(x12,solver)
        sigma1=alg.dirneu(dirich,neum,b11,b22,lmda1,lmda2,f1,f2, maxit = mit, tol = tl, show = False, all_gamma = True,mute = True)

        exact=esol(x11,x12,b11,b22,lmda1,lmda2,f1,f2)[0](gx)
        k1+=  get_largest_div(sigma1[2],exact,conv_rate)
    k1 = k1/tests

    k2 = 0
    for i in range(tests):

        if(dirEqui):
            x21 = sc.linspace(x1,gx,n1+2)        
        else:
            x21 = np.zeros(n1+2)
            x21[0] = x1
            x21[1:-1] = x1+(gx-x1)*np.random.rand(n1)
            x21[-1] = gx
            x21.sort()

        if(neuEqui):
            x22 = sc.linspace(gx,x2,n2+2)
        else:
            x22 = np.zeros(n2+2)
            x22[0] = gx
            x22[1:-1] = gx+(x2-gx)*np.random.rand(n2)
            x22[-1] = x2
            x22.sort()

        dirich = discdir(x21,solver)
        neum = discneu(x22,solver)
        sigma2=alg.dirneu(dirich,neum,b11,b22,lmda1,lmda2,f1,f2, maxit = mit, tol = tl, show = False, all_gamma = True,mute = True)
        exact=esol(x21,x22,b11,b22,lmda1,lmda2,f1,f2)[0](gx)
        k2+=  get_largest_div(sigma2[2],exact,conv_rate)
    k2 = k2/tests

    k3 = 0
    for i in range(tests):
        if(dirEqui):
            x31 =sc.linspace(x1,gx,n1+2) 
        else:
            x31 = np.zeros(n1+2)
            x31[0] = x1
            x31[1:-1] = x1+(gx-x1)*Ldense(np.random.rand(n1))
            x31[-1] = gx
            x31.sort()
        if(neuEqui):
            x32 = sc.linspace(gx,x2,n2+2)
        else:
            x32 = np.zeros(n2+2)
            x32[0] = gx
            x32[1:-1] = gx+(x2-gx)*Rdense(np.random.rand(n2))
            x32[-1] = x2
            x32.sort()
        dirich = discdir(x31,solver)
        neum = discneu(x32,solver)
        sigma3=alg.dirneu(dirich,neum,b11,b22,lmda1,lmda2,f1,f2, maxit = mit, tol = tl, show = False, all_gamma = True,mute = True)
        
        exact=esol(x31,x32,b11,b22,lmda1,lmda2,f1,f2)[0](gx)
        k3+=  get_largest_div(sigma3[2],exact,conv_rate)
    k3 = k3/tests

    k4 = 0
    for i in range(tests):
        if(dirEqui):
            x41 =sc.linspace(x1,gx,n1+2) 
        else:
            x41 = np.zeros(n1+2)
            x41[0] = x1
            x41[1:-1] = x1+(gx-x1)*Ldense(np.random.rand(n1))
            x41[-1] = gx
            x41.sort()
        if(neuEqui):
            x42 = sc.linspace(gx,x2,n2+2)
        else:
            x42 = np.zeros(n2+2)
            x42[0] = gx
            x42[1:-1] = gx+(x2-gx)*Rdense(np.random.rand(n2))
            x42[-1] = x2
            x42.sort()
        dirich = discdir(x41,solver)
        neum = discneu(x42,solver)
        sigma4=alg.dirneu(dirich,neum,b11,b22,lmda1,lmda2,f1,f2, maxit = mit, tol = tl, show = False, all_gamma = True,mute = True)
        exact=esol(x41,x42,b11,b22,lmda1,lmda2,f1,f2)[0](gx)
        k4+=  get_largest_div(sigma4[2],exact,conv_rate)
    k4 = k4/tests
    return [k1,k2,k3,k4]



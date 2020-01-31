import scipy as sc
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import Approx as fa
import DefaultSolver as ds
class FVM:
    def __init__(self,x_grid,solver=ds.SPSolver()):
        self.x_grid = x_grid
        self.solver = solver
    def dirdisc(self,f,lmda,b1,b2):
        n=len(self.x_grid)-2
        A = sc.sparse.lil_matrix((n,n))
        for i in range(n):
            A[i,i] = (lmda/(self.x_grid[i+1]-self.x_grid[i])) + (lmda/(self.x_grid[i+2]-self.x_grid[i+1])) 
            if(i+1<n):
                A[i,i+1] = -lmda/(self.x_grid[i+2]-self.x_grid[i+1])
            if(i-1>= 0):
                A[i,i-1] = -lmda/(self.x_grid[i+1]-self.x_grid[i])
        B = sc.zeros(n)
        for i in range(n):
            xm = self.x_grid[i+1]-0.5*(self.x_grid[i+1]-self.x_grid[i])
            xp = self.x_grid[i+2]-0.5*(self.x_grid[i+2]-self.x_grid[i+1])
            B[i] -= integrate.quad(f,xm,xp)[0]
        B[0] = B[0]+b1*(lmda/(self.x_grid[1]-self.x_grid[0]))
        B[-1] = B[-1]+b2*(lmda/(self.x_grid[-1]-self.x_grid[-2]))
        return self.solver.solve(A,B)

    def neudisc(self,f,lmda,bprime,b2):
        n=len(self.x_grid)-2
        A = sc.sparse.lil_matrix((n+1,n+1))
        for i in range(n):
            A[i,i] = lmda*(1./(self.x_grid[i+1]-self.x_grid[i])) + lmda*(1./(self.x_grid[i+2]-self.x_grid[i+1])) 
            if(i+1<n):
                A[i,i+1] = -lmda/(self.x_grid[i+2]-self.x_grid[i+1])
            if(i-1>= 0):
                A[i,i-1] = -lmda/(self.x_grid[i+1]-self.x_grid[i])
        B = sc.zeros(n+1)
        for i in range(n):
            xm = self.x_grid[i+1]-0.5*(self.x_grid[i+1]-self.x_grid[i])
            xp = self.x_grid[i+2]-0.5*(self.x_grid[i+2]-self.x_grid[i+1])
            B[i] -= integrate.quad(f,xm,xp,epsabs=1e-14,epsrel=1e-14)[0]
        A[0,-1]= -(lmda/(self.x_grid[1]-self.x_grid[0]))
    
        dx1 = self.x_grid[1]-self.x_grid[0]
        A[-1,-1] = -lmda/dx1
        A[-1,0] = lmda/dx1
        B[-1]  = lmda*bprime+integrate.quad(f,self.x_grid[0],self.x_grid[0]+0.5*(self.x_grid[1]-self.x_grid[0]))[0]    
        B[-2] += lmda*(b2/(self.x_grid[-1]-self.x_grid[-2]))
        return self.solver.solve(A,B)

    """
    Solve the dirichlet problem
    """
    def dir(self,f,lmda,b1,b2):
        C=self.dirdisc(f,lmda,b1,b2)
        return fa.linapprox(self.x_grid,b1,b2,C)
    """
    Solve the dirichlet problem and get a second order
    approximation of the derivitive at the second boundary
    point
    """
    def dirlastder(self,f,lmda,b1,b2):
        C=self.dirdisc(f,lmda,b1,b2)
        intpart = (1./lmda)*integrate.quad(lambda x: f(x),self.x_grid[-2]+0.5*(self.x_grid[-1]-self.x_grid[-2]),self.x_grid[-1])[0]
        return (b2-C[-1])/(self.x_grid[-1]-self.x_grid[-2])+intpart

    """
    Solve the neumann problem
    """
    def neu(self,f,lmda,b1p,b):
        C=self.neudisc(f,lmda,b1p,b)
        return fa.linapprox(self.x_grid,C[-1],b,C[:-1])
    """
    Get the function value at the first boundary possition
    """
    def neufirstval(self,f,lmda,b1p,b):
        C=self.neudisc(f,lmda,b1p,b)
        return C[-1]


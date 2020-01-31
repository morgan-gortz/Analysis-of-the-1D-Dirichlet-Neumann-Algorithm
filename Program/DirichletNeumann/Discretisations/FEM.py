import scipy as sc
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import Approx as fa
import DefaultSolver as ds
def phi(i,x,x_grid):
    if(x<=x_grid[0]): return 0;
    if(x>=x_grid[-1]): return 0;
    if(not(x_grid[i]<x<x_grid[i+2])): return 0;
    if(x_grid[i]< x <= x_grid[i+1]):
        return (x-x_grid[i])/(x_grid[i+1]-x_grid[i])
    if(x_grid[i+1]< x < x_grid[i+2]): 
        return (x_grid[i+2]-x)/(x_grid[i+2]-x_grid[i+1])
    return 0;

def der_phi(i,x,x_grid):
    if(x<=x_grid[0]): return 0;
    if(x>=x_grid[-1]): return 0;
    if(not(x_grid[i]<x<x_grid[i+2])): return 0;
    if(x_grid[i]< x <= x_grid[i+1]):
        return (1)/(x_grid[i+1]-x_grid[i])
    if(x_grid[i+1]< x < x_grid[i+2]): 
        return (-1)/(x_grid[i+2]-x_grid[i+1])
    return 0;
def phi_start(x,x_grid):
    if(x<x_grid[0]): return 0
    if(x>=x_grid[1]): return 0
    return (x_grid[1]-x)/(x_grid[1]-x_grid[0])

def phi_end(x,x_grid):
    if(x<x_grid[-2]): return 0
    if(x>x_grid[-1]): return 0
    return (x-x_grid[-2])/(x_grid[-1]-x_grid[-2])



class FEM:
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
            B[i] -= integrate.quad(lambda x: f(x)*phi(i,x,self.x_grid),self.x_grid[i],self.x_grid[i+2])[0]
        B[0] = B[0]+b1*(lmda/(self.x_grid[1]-self.x_grid[0]))
        B[-1] = B[-1]+b2*(lmda/(self.x_grid[-1]-self.x_grid[-2]))
        return self.solver.solve(A,B)


    def neudisc(self,f,lmda,b2p,b):
        n = len(self.x_grid)-2
        A = sc.sparse.lil_matrix((n+1,n+1))
        for i in range(n):
            A[i,i] = (lmda/(self.x_grid[i+1]-self.x_grid[i]))+(lmda/(self.x_grid[i+2]-self.x_grid[i+1]))
            if(i+1<n):
                A[i,i+1] = -lmda/(self.x_grid[i+2]-self.x_grid[i+1])
            if(i-1>= 0):
                A[i,i-1] = -lmda/(self.x_grid[i+1]-self.x_grid[i])
        A[0,-1] = -lmda/(self.x_grid[1]-self.x_grid[0])
        
        A[-1,0] = 1./(self.x_grid[1]-self.x_grid[0])
        A[-1,-1] = -1./(self.x_grid[1]-self.x_grid[0])
        B = sc.zeros(n+1)
        for i in range(n):
            B[i] = -integrate.quad(lambda x: f(x)*phi(i,x,self.x_grid),self.x_grid[i],self.x_grid[i+2])[0]
        B[-2] = B[-2]+b*(lmda/(self.x_grid[-1]-self.x_grid[-2]))
        B[-1] = b2p+(1./lmda)*integrate.quad(lambda x: f(x)*phi_start(x,self.x_grid),self.x_grid[0],self.x_grid[1])[0]
        return self.solver.solve(A,B)

    """
    Solve the dirichlet problem
    """
    def dir(self,f,lmda,b1,b2):
        C=self.dirdisc(f,lmda,b1,b2)
        return fa.femapprox(self.x_grid,b1,b2,C)
    """
    Solve the dirichlet problem and get a second order
    approximation of the derivitive at the second boundary
    point
    """
    def dirlastder(self,f,lmda,b1,b2):
        C=self.dirdisc(f,lmda,b1,b2)
        intpart = (1./lmda)*integrate.quad(lambda x: f(x)*phi_end(x,self.x_grid),self.x_grid[-2],self.x_grid[-1])[0]
        return (b2-C[-1])/(self.x_grid[-1]-self.x_grid[-2])+intpart

    """
    Solve the neumann problem
    """
    def neu(self,f,lmda,b1p,b):
        C=self.neudisc(f,lmda,b1p,b)
        return fa.femapprox(self.x_grid,C[-1],b,C[:-1])
    """
    Get the function value at the first boundary possition
    """
    def neufirstval(self,f,lmda,b1p,b):
        C=self.neudisc(f,lmda,b1p,b)
        return C[-1]


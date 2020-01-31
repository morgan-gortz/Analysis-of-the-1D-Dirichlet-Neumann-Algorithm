import scipy as sc
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import Approx as fa
import DefaultSolver as ds
class FDM:
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
            B[i] -= (self.x_grid[-1]-self.x_grid[-2])*f(self.x_grid[i+1])
        B[0] = B[0]+b1*(lmda/(self.x_grid[1]-self.x_grid[0]))
        B[-1] = B[-1]+b2*(lmda/(self.x_grid[-1]-self.x_grid[-2]))
        return self.solver.solve(A,B)


    """
    The basic neuman discretistation
    """
    def neudisc(self,f,lmda,b1p,b):
        n = len(self.x_grid)-2
        A = sc.sparse.lil_matrix((n+1,n+1))
        for i in range(n):
            A[i,i] = (lmda/(self.x_grid[i+1]-self.x_grid[i]))+(lmda/(self.x_grid[i+2]-self.x_grid[i+1]))
            if(i+1<n):
                A[i,i+1] = -lmda/(self.x_grid[i+2]-self.x_grid[i+1])
            if(i-1>= 0):
                A[i,i-1] = -lmda/(self.x_grid[i+1]-self.x_grid[i])
        A[0,-1] = -lmda/(self.x_grid[1]-self.x_grid[0])
        
        A[-1,1] = -1./(2*(self.x_grid[1]-self.x_grid[0]))
        A[-1,0] = 4./(2*(self.x_grid[1]-self.x_grid[0]))
        A[-1,-1] = -3./(2*(self.x_grid[1]-self.x_grid[0]))

        B = sc.zeros(n+1)
        for i in range(n):
            B[i] = -(self.x_grid[-1]-self.x_grid[-2])*f(self.x_grid[i+1])
        B[-2] = B[-2]+b*(lmda/(self.x_grid[-1]-self.x_grid[-2]))
        B[-1] = b1p
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
        return (3*b2-4*C[-1]+C[-2])/(2*(self.x_grid[1]-self.x_grid[0]))

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

import scipy as sc
import FEM as fem

def femapprox(x_grid,b1,b2,p):
    def f(x):
        res = b1*fem.phi_start(x,x_grid)
        res += b2*fem.phi_end(x,x_grid)
        for i in range(len(x_grid)-2):
            res+=p[i]*fem.phi(i,x,x_grid)
        return res
    return f

"""
Return a piecwise linear function defined by a grid and interpolation points
[b1,p[0],..,p[n],b2]
"""
def linapprox(x_grid,b1,b2,p): 
    def f(x):
        if(x_grid[0]>x or x>x_grid[-1]):
            return 0
        if(x_grid[0]<=x<x_grid[1]):
            return b1*(x_grid[1]-x)/(x_grid[1]-x_grid[0]) + p[0]*(x-x_grid[0])/(x_grid[1]-x_grid[0])
        if(x_grid[-2]<=x<=x_grid[-1]):
            return p[-1]*(x_grid[-1]-x)/(x_grid[-1]-x_grid[-2]) + b2*(x-x_grid[-2])/(x_grid[-1]-x_grid[-2])
        for i in range(1,len(p)):
            if(x_grid[i]<=x<x_grid[i+1]):
               return p[i-1]*(x_grid[i+1]-x)/(x_grid[i+1]-x_grid[i]) + p[i]*(x-x_grid[i])/(x_grid[i+1]-x_grid[i])
    return f 


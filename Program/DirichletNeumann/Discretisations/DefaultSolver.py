import scipy.sparse.linalg as slin
import scipy as sc

class SPSolver:
    def solve(self,A,b):
        return slin.spsolve(A,b)

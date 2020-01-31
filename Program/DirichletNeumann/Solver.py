import scipy.sparse.linalg as slin
import scipy.sparse as sp
import scipy as sc

class SPSolver:
    def solve(self,A,b):
        return slin.spsolve(sp.csr_matrix(A),b)

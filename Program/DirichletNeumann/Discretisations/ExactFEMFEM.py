import scipy as sc
import scipy.sparse as sp
import Approx as ap
import scipy.sparse.linalg as slin
import scipy.integrate as integrate
import FEM as fem

def Agen(A,ox,oy,xs,lmd):
    n = len(xs)-2
    for i in range(n):
        if(i!=0):
           A[i+ox,i-1+oy]=-lmd/(xs[i+1]-xs[i])
        A[i+ox,i+oy]=lmd/(xs[i+1]-xs[i])+ lmd/(xs[i+2]-xs[i+1])
        if(i!=n-1):
            A[i+ox,i+1+oy]=-lmd/(xs[i+2]-xs[i+1])

def exactFEMFEM(x1,x2,ys,ye,lmd1,lmd2,f1,f2):
    n1 = len(x1)-2
    n2 = len(x2)-2
    A = sp.lil_matrix((n1+n2+1, n1+n2+1))
    Agen(A,0,0,x1,lmd1)
    Agen(A,n1,n1,x2,lmd2)
    
    A[n1-1,-1]=-lmd1/(x1[-1]-x1[-2])
    #add second system
    A[n1,-1]=-lmd2/(x2[1]-x2[0])
    #add dirvitive equation
    A[-1,n1-1]= -lmd1/(x1[-1]-x1[-2])
    A[-1,n1] = -lmd2/(x2[1]-x2[0])
    A[-1,-1] = lmd1/(x1[-1]-x1[-2])+lmd2/(x2[1]-x2[0])
    B = sc.zeros(n1+n2+1)
    B[0]=ys*(lmd1/(x1[1]-x1[0]))
    B[-2]=ye*(lmd2/(x2[-1]-x2[-2]))
    for i in range(n1):
        B[i] -= integrate.quad(lambda x: f1(x)*fem.phi(i,x,x1),x1[i],x1[i+2])[0]
    for i in range(n2):
        B[n1+i] -= integrate.quad(lambda x: f2(x)*fem.phi(i,x,x2),x2[i],x2[i+2])[0]
    B[-1]-= integrate.quad(lambda x: f2(x)*fem.phi_start(x,x2),x2[0],x2[1])[0]
    B[-1]-= integrate.quad(lambda x: f1(x)*fem.phi_end(x,x1),x1[-2],x1[-1])[0]

    sol = list(slin.spsolve(sp.csr_matrix(A),B))
    gamma = sol[-1]
    vh = ap.femapprox(x1,ys,gamma,sol[:n1])
    wh = ap.femapprox(x2,gamma,ye,sol[n1:n2+n1])
    return (vh,wh)

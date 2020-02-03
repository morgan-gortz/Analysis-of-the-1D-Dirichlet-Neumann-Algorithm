import scipy as sc
import DirichletNeumann as dn
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
################################# USER SETTINGS ###################################

"""
Diffeq params
"""
lmd1 = 1.0
lmd2 = 2.0
xa = 0.0
xg = 1.0
xb = 2.0
ua = 1.0
ub = 3.0
f1 = lambda x: sc.sin(x)
f2 = lambda x: sc.cos(3*x)
globres = 1000 #Resoltion of global discretisations
dnres1 = 5 # Resolution of the dir/neu alg discretisations
dnres2 = 7
"""
Discretisation params
"""
dirDisc = "FEM" # FDM,FEM,FVM
neuDisc = "FVM" # FDM,FEM,FVM


#############################  PROGRAM ##############################

def Ldense(x):
    return sc.expm1(3*x)/(sc.exp(3)-1)

def Rdense(x):
    return (sc.log(x*(sc.e-1)+1))



"""
Generate an approximation using global discretisation
"""
gx1 = sc.linspace(xa,xg,globres)
gx2 = sc.linspace(xg,xb,globres)
gsol = dn.efemfem.exactFEMFEM(gx1,gx2,ua,ub,lmd1,lmd2,f1,f2)
"""
Generate an approximation using the dir/neu algorithm 
"""

if(dirDisc=="FDM"):
    discdir = dn.fdm.FDM
elif(dirDisc=="FEM"):
    discdir = dn.fem.FEM
elif(dirDisc=="FVM"):
    discdir = dn.fvm.FVM
else:
    raise RuntimeError("Discretisations have to be FDM,FEM,FVM")

if(neuDisc=="FDM"):
    discneu = dn.fdm.FDM
elif(neuDisc=="FEM"):
    discneu = dn.fem.FEM
elif(neuDisc=="FVM"):
    discneu = dn.fvm.FVM
else:
    raise RuntimeError("Discretisations have to be FDM,FEM,FVM")
plt.plot(gx1,[gsol[0](x) for x in gx1],'k--',label = r'$u(x)$')
plt.plot(gx2,[gsol[1](x) for x in gx2], 'k--')
for i in range(1,5):
    dnx1 = sc.linspace(0,1,dnres1)
    dnx1 = [Rdense(x) for x in dnx1]
    dnx1.sort()
    dnx1 = [xa+x*(xg-xa) for x in dnx1]
    
    dnx2 = sc.linspace(0,1,dnres2)
    dnx2 = [Ldense(x) for x in dnx2]
    dnx2.sort()
    dnx2 = [xg+x*(xb-xg) for x in dnx2]
    
    dnsol = dn.dirneu(discdir(dnx1),discneu(dnx2),ua,ub,lmd1,lmd2,f1,f2,show = False,maxit = i)
    plt.plot(dnx1,[dnsol[0](x) for x in dnx1],'k-', label = r'$u_{itr}(x), \ itr = 1,..,4$',marker = '*')
    plt.plot(dnx2,[dnsol[1](x) for x in dnx2],'k-', marker = '*')
plt.plot([xg],[gsol[0](xg)],'ko', label = r'$x_\Gamma$')
plt.legend()
plt.show()


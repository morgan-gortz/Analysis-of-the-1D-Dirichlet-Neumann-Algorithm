import scipy as sc
import DirichletNeumann as dn
import matplotlib.pyplot as plt

################################# USER SETTINGS ###################################

"""
Diffeq params
"""
lmd1 = 1.0
lmd2 = 4.0
xa = 0.0
xg = 1.2
xb = 2.0
ua = 4.0
ub = 3.0
f1 = lambda x: x
f2 = lambda x: -x
globres = 1000 #Resoltion of global discretisations
dnres = 5 # Resolution of the dir/neu alg discretisations

"""
Discretisation params
"""
dirDisc = "FEM" # FDM,FEM,FVM
neuDisc = "FEM" # FDM,FEM,FVM


#############################  PROGRAM ##############################

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
dnx1 = sc.linspace(xa,xg,dnres)
dnx2 = sc.linspace(xg,xb,dnres)

dnsol = dn.dirneu(discdir(dnx1),discneu(dnx2),ua,ub,lmd1,lmd2,f1,f2,show = False)



plt.plot(gx1,[gsol[0](x) for x in gx1],'r--',label = "u glob")
plt.plot(gx2,[gsol[1](x) for x in gx2], 'r--')
plt.plot(dnx1,[dnsol[0](x) for x in dnx1],'b-.', label = "u dirneu")
plt.plot(dnx2,[dnsol[1](x) for x in dnx2],'b-.')
plt.legend()

plt.show()


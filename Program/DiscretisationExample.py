import DirichletNeumann as dn
import scipy as sc
import matplotlib.pyplot as plt
"""
This script shows how the discretisations are used
"""

discType = "FVM" # FDM,FEM,FVM
boundaryType = "NEUDIR" # NEUDIR, DIRDIR
grid = sc.linspace(0,1,100)
f = lambda x: -2
a = 1 # left bc (Neumann in NEUDIR)
b = 2 # right bc (always Dirichlet)
lmda = 2 # lambda in the diff equation

linSolver = dn.SPSolver()
if(discType == "FDM"):
    disc = dn.fdm.FDM(grid,linSolver)
elif(discType == "FEM"):
    disc = dn.fem.FEM(grid,linSolver)
elif(discType == "FVM"):
    disc = dn.fvm.FVM(grid,linSolver)
else:
    raise RuntimeError("Discretisation has to be FDM,FEM or FVM")
if(boundaryType == "DIRDIR"):
    u = disc.dir(f,lmda,a,b)
elif(boundaryType == "NEUDIR"):
    u = disc.neu(f,lmda,a,b)
else:
    raise RuntimeError("boundaryType has to be DIRDIR or NEUDIR")
#Plot the solution
plt.plot(grid,[u(x) for x in grid])
plt.show()

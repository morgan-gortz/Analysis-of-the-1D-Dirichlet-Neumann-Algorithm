import scipy as sc
def dirneu(dirich,neumann,b11,b22,lmda1,lmda2,f1,f2, maxit = 100, tol = 1e-8, show = True, all_gamma=True,mute = False):
    it = 0
    gammak  = 0
    if(all_gamma):
        gammas = [gammak]
    while(it<maxit):
        der = (lmda1/lmda2)*dirich.dirlastder(f1,lmda1,b11,gammak);
        gammakp1 = neumann.neufirstval(f2, lmda2,der,b22)
        if(abs(gammak-gammakp1)<tol):
            gammak = gammakp1
            break;
        it+= 1
        gammak = gammakp1
        if(all_gamma):
            gammas.append(gammak)

    if(maxit == it and (not mute)): 
        print("!!!! ERROR DID NOT CONVERGE !!!!");
    u1 = dirich.dir(f1,lmda1,b11,gammak);
    u2 = neumann.neu(f2, lmda2,der,b22)
    if(show):
        x1 = sc.linspace(dirich.x_grid[0],dirich.x_grid[-1],100)
        x2 = sc.linspace(neumann.x_grid[0],neumann.x_grid[-1],100)
        plt.plot(x1,[u1(x) for x in x1])
        plt.plot(x2,[u2(x) for x in x2])
        plt.show()
    if(all_gamma):
        return [u1,u2,gammas]
    else:
        return (u1,u2)


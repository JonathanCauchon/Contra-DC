
from Modules import *

from mpl_toolkits.mplot3d import Axes3D


sims = np.loadtxt("Interpolation/indices.txt")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("Wavelength (nm)")
ax.set_zticks((1400, 1500, 1600, 1700))
ax.set_xticks((0.55, 0.56, 0.57))
ax.set_yticks((0.43, 0.44, 0.45))
ax.set_xlabel("Wg1 Width (um)")
ax.set_ylabel("Wg2 Width (nm)")
img = ax.scatter(sims[:,0]*1e6, sims[:,1]*1e6, sims[:,2]*1e9, c=sims[:,3], cmap=plt.plasma())
cbar = plt.colorbar(img)
cbar.set_label("neff")
ax.view_init(10,20)
plt.show()
fig.savefig("3D_interpolation.pdf",bbox_inches='tight')
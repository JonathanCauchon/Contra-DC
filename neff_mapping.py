
from Modules import *

from mpl_toolkits.mplot3d import Axes3D

sims = np.loadtxt("Interpolation/indices.txt")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

img = ax.scatter(sims[:,0]*1e6, sims[:,1]*1e6, sims[:,2]*1e9, c=sims[:,3],cmap=plt.hot())
fig.colorbar(img)
plt.xlabel("Width of wg1 (um)")
plt.ylabel("Width of wg2 (um)")
# plt.zlabel("Wavelength (nm)")
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = np.random.standard_normal(100)
# y = np.random.standard_normal(100)
# z = np.random.standard_normal(100)
# c = np.random.standard_normal(100)

# img = ax.scatter(x, y, z, c, cmap=plt.hot())
# fig.colorbar(img)
# # plt.show()
# print(c)
# print(wvl)
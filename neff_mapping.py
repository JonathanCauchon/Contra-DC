
from Modules import *

from mpl_toolkits.mplot3d import Axes3D

n1 = np.reshape(np.loadtxt("Database/neff/neff_1.txt"),(5,5,5))
n2 = np.reshape(np.loadtxt("Database/neff/neff_2.txt"),(5,5,5))
w1, w2, wvl = np.loadtxt("Database/neff/w1_w2_lambda.txt")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

img = ax.scatter(w1*1e6, w2*1e6, wvl*1e9, c=n1[0,0,:],cmap=plt.hot())
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
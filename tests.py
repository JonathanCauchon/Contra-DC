from Modules import *

m = np.loadtxt("indices.txt")

w1 = 5.55e-7
w2 = 4.35e-7
n = 100
wavelength = np.linspace(1.4e-6,1.7e-6,n)
idx_w1 = np.where(m==w1,1,0)[:,0]
# idx_w1 = np.where(m==w1,1,0)[0]
# idx_w2 = np.where(m==w2,1,0)[0]
idx_w2 = np.where(m==w2,1,0)[:,1]
print(idx_w1)
print(idx_w2)
idx = np.where(idx_w1*idx_w2==1)
m_ = m[idx]
print(idx_w1)





p1, p2 = np.polyfit(m_[:,2], m_[:,3],2), np.polyfit(m_[:,2], m_[:,4],2)
n1 = p1[0]*wavelength**2 + p1[1]*wavelength + p1[2]
n2 = p2[0]*wavelength**2 + p2[1]*wavelength + p2[2]

plt.figure()
plt.plot(m_[:,2]*1e9, m_[:,3],"o-",label="Simulated")
plt.plot(wavelength*1e9,n1,".",label="Fitted")
plt.legend()
plt.show()

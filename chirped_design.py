from modules import *
from ContraDC import *

figsize = np.array([8.5,11/4])*2
fig = plt.figure(figsize=figsize)
grid = plt.GridSpec(4,3)
plt.subplots_adjust(hspace=.4, wspace=.6)


""" temp chirp drop response """

if 1:

    ax = plt.subplot(grid[2:,2])

    d = ContraDC(N=1200, apod_shape="tanh", period=314.7e-9, 
                wvl_range=[1500e-9,1600e-9], resolution=1000, alpha=0)
    d.T_profile = np.linspace(300,443,100)
    d.simulate()

    l_seg = int(d.N/d.N_seg)*2
    num_curves = 49

    # colors
    r = np.flip(np.linspace(1,0,num_curves))
    g = np.zeros(r.shape)
    b = np.flip(np.linspace(0,1,num_curves))

    for i, j in zip(np.linspace(2,98,num_curves), range(r.shape[0])):
        i = int(i)

        d2 = ContraDC(N=1200 - i*12, N_seg=100-i, apod_shape="tanh", period=314.7e-9, 
                    wvl_range=[1500e-9,1600e-9], resolution=1000, alpha=0)

        d2.T_profile = d.T_profile[:-i]
        d2.apod_profile = d.apod_profile[:-i]
        d2.getChirpProfile().getPropConstants().propagate()

        ax.plot(d2._wavelength, d2.drop, c=[r[j], g[j], b[j]], alpha=.75)

    ax.plot(d._wavelength, d.drop, "k", lw=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Drop (nm)")
    ax.set_ylim((-70,1))


if 1:
    ax = plt.subplot(grid[2:,1])
    d = ContraDC(N=1200, T=[300,600]).simulate()
    z = 0
    print(d.T_profile.shape)
    r = np.linspace(0,1,99)
    g = np.zeros(r.shape)
    b = np.linspace(1,0,100)
    for i, seg in enumerate(range(d.N_seg-1)):
        print(i)
        ax.plot(320e-3*np.asarray([z, z + d.N/d.N_seg]), [d.T_profile[i], d.T_profile[i+1]], c=[r[i], g[i], b[i]])
        z += d.N/d.N_seg
        ax1 = ax.twinx()
        ax1.set_ylim((1538,1559))
        ax1.set_yticks((1540,1545,1550,1555))
        ax1.set_ylabel("$\lambda_B$ (nm)", rotation=-90, labelpad=15)
        # ax1.plot()
    ax.set_ylabel("Temperature (K)")
    ax.set_xlabel("z ($\mu$m)")


if 1:
    ax = plt.subplot(grid[0,1])
    d = ContraDC(N=1200, w1=[.55e-6,.57e-6], w2=[.43e-6,.45e-6], w_chirp_step=.5e-9).simulate()
    z = 0
    print(d.T_profile.shape)
    r = np.linspace(0,0,99)
    b = np.linspace(0,0,99)
    g = np.linspace(1,.2,99)
    # r = np.append(np.linspace(0,1,50), np.linspace(1,1,50))
    # g = np.append(np.linspace(1,1,50), np.linspace(1,0,50))
    # b = np.linspace(0,0,100)
    # lw = np.linspace(.5,2,99)
    for i, seg in enumerate(range(d.N_seg-1)):
        print(i)
        ax.plot(320e-3*np.asarray([z, z + d.N/d.N_seg]), [d.w1_profile[i]*1e6, d.w1_profile[i+1]*1e6], c=[r[i], g[i], b[i]])
        z += d.N/d.N_seg
        ax1 = ax.twinx()
        ax1.set_ylim((1538,1559))
        ax1.set_yticks((1540,1545,1550,1555))
        ax1.set_ylabel("$\lambda_B$ (nm)", rotation=-90, labelpad=15)
        # ax1.plot()
    ax.set_xticklabels(())
    ax.set_ylabel("w1 ($\mu$m)")
    ax.set_xlabel("z ($\mu$m)")


if 1:
    ax = plt.subplot(grid[1,1])
    d = ContraDC(N=1200, w1=[.55e-6,.57e-6], w2=[.43e-6,.45e-6], w_chirp_step=.5e-9).simulate()
    z = 0
    print(d.T_profile.shape)
    r = np.linspace(0,0,99)
    b = np.linspace(0,0,99)
    g = np.linspace(1,.2,99)
    # r = np.append(np.linspace(0,1,50), np.linspace(1,1,50))
    # g = np.append(np.linspace(1,1,50), np.linspace(1,0,50))
    # b = np.linspace(0,0,100)
    # lw = np.linspace(.5,2,99)
    for i, seg in enumerate(range(d.N_seg-1)):
        print(i)
        ax.plot(320e-3*np.asarray([z, z + d.N/d.N_seg]), [d.w2_profile[i]*1e6, d.w2_profile[i+1]*1e6], c=[r[i], g[i], b[i]])
        z += d.N/d.N_seg
        ax1 = ax.twinx()
        ax1.set_ylim((1538,1559))
        ax1.set_yticks((1540,1545,1550,1555))
        ax1.set_ylabel("$\lambda_B$ (nm)", rotation=-90, labelpad=15)
        # ax1.plot()
    ax.set_ylabel("w2 ($\mu$m)")
    ax.set_xlabel("z ($\mu$m)")


if 1:

    ax = plt.subplot(grid[0:2,2])

    d = ContraDC(N=1200, apod_shape="tanh", period=314.7e-9, 
                wvl_range=[1500e-9,1600e-9], resolution=1000, alpha=0,
                w1=[.55e-6,.57e-6], w2=[.43e-6,.45e-6], w_chirp_step=.01e-9)
    # d.T_profile = np.linspace(300,443,100)
    d.simulate()

    l_seg = int(d.N/d.N_seg)*2
    num_curves = 49

    # colors
    r = np.linspace(0,0,99)
    b = np.linspace(0,0,99)
    g = np.linspace(.2,1,50)

    # r = np.flip(np.append(np.linspace(0,1,25), np.linspace(1,1,25)))
    # g = np.flip(np.append(np.linspace(1,1,25), np.linspace(1,0,25)))
    # b = np.linspace(0,0,50)

    for i, j in zip(np.linspace(2,98,num_curves), range(r.shape[0])):
        i = int(i)

        d2 = ContraDC(N=1200 - i*12, N_seg=100-i, apod_shape="tanh", period=314.7e-9, 
                    wvl_range=[1500e-9,1600e-9], resolution=1000, alpha=0)

        d2.w1_profile = d.w1_profile[:-i]
        d2.w2_profile = d.w2_profile[:-i]
        d2.apod_profile = d.apod_profile[:-i]
        d2.getChirpProfile().getPropConstants().propagate()

        ax.plot(d2._wavelength, d2.drop, c=[r[j], g[j], b[j]], alpha=.75, zorder=100-i)

    ax.plot(d._wavelength, d.drop, "k", lw=2, zorder=150)
    ax.set_ylabel("drop (nm)")
    ax.set_xlabel("Wavelength (nm)")
    # ax.set_xticklabels(())
    ax.set_ylim((-70,1))




if 1:
    ax = plt.subplot(grid[:,0])

    d = ContraDC(T=300, wvl_range=[1500e-9, 1600e-9], period=314e-9).simulate()
    d2 = ContraDC(T=600, wvl_range=[1500e-9, 1600e-9], period=314e-9).simulate()
    alpha=.5
    ax.fill_between(d._wavelength, d.n1_profile[:,50], d2.n1_profile[:,50], facecolor="red", alpha=alpha, label="n_{eff, 1}")
    ax.fill_between(d._wavelength, d.n2_profile[:,50], d2.n2_profile[:,50], facecolor="b", alpha=alpha)
    ax.fill_between(d._wavelength, (d.n1_profile[:,50] + d.n2_profile[:,50])/2, (d2.n1_profile[:,50] + d2.n2_profile[:,50])/2, facecolor=[.5,0,.5], alpha=alpha)
    p1 = [1538.13, 2.44926]
    p2 = [1559.02, 2.48252]
    prop = dict(arrowstyle="-|>,head_width=.5,head_length=1", fc="k", ec="k", lw=4, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=p2, xytext=p1, arrowprops=prop)
    ax.scatter( 0 ,0,marker="$\Rightarrow$", label="Chirp Sweep" )
    ax.plot([p1[0], p1[0]], [p1[1], 0], "k--")
    ax.plot([p2[0], p2[0]], [p2[1], 0], "k--")
    # plt.arrow(p1[0], p1[1], p2[0]-p1[0]-3, p2[1]-p1[1], head_length=3, head_width=.03, lw=4, facecolor="r", ec="r", )
    # plt.plot([d.wavelength/2/d.period])
    # print(np.where)
    ax.plot(d._wavelength, d.wavelength/2/d.period, "k--")
    # plt.legend()
    ax.text(1540,2.29,"$\lambda_{1}$", fontsize=16)
    ax.text(1562,2.29,"$\lambda_{2}$", fontsize=16)
    ax.text(1520,2.56, "$n_{eff, 1}$", fontsize=18, rotation=-10)
    ax.text(1520,2.47, "$n_{eff}}$", fontsize=18, rotation=-10)
    ax.text(1520,2.375, "$n_{eff, 2}$", fontsize=18, rotation=-10)
    ax.plot([1522,1528],[2.495,2.488], "k")
    ax.set_ylim((np.min(d.n2_profile[:,50]),np.max(d2.n1_profile[:,50])))
    ax.set_xlim((1500,1600))



plt.show()
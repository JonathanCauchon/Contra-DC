# Contra-DC
fully parameterized contra-directional coupler model including pitch and width chirp.
offers to create fully parameterizable cdc object and simulate response with TMM method.

- Stable version: ChirpedContraDC_v2.py
- v3 incorporates tools for chirp optimization (under dev.)
- v4 incorporates exportGDSInfo method to easily create layouts of simulated devices

See example below for basic usage.

```python
# Example of ChirpedContraDC_v3 class usage
# Many more optional properties inside class declaration

from Modules import *
from ChirpedContraDC_v3 import *


# 3 customizable linear chirps
w1 = [.55e-6, .551e-6]
w2 = [.44e-6, .441e-6]
p = [324e-9, 326e-9]

# wvl range to plot
wr = [1530e-9, 1600e-9] 

# Device creation, simulation and performance assessment
device = ChirpedContraDC(N=2500, a=10, w1=w1, w2=w2, period=p, wvl_range=wr, resolution=300, N_seg=100)
device.simulate()
device.displayResults()
```
![alt text](https://github.com/JonathanCauchon/Contra-DC/blob/master/ANT_Nov_2019/Figure_1.png "Result of simulation")


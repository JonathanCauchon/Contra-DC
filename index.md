# Contra-DC

Fully parameterizable contra-directional coupler model including chirp.
Offers to create fully parameterizable CDC object and simulate response with TMM method. 

- See [the documentation](https://github.com/JonathanCauchon/Contra-DC/tree/master/Documentation) for details on the physics of the device.


### Installation
```sh
git clone https://github.com/JonathanCauchon/Contra-DC
```
#### ContraDC.ContraDC.simulate()
	:attributes: Hi

See example below for basic usage.



```python
# Example of ChirpedContraDC_v7 class usage
# Many more optional properties inside class definition

from ChirpedContraDC_v7 import *

# grating parameters
w1 = .56e-6 # waveguide 1 width
w2 = .44e-6 # waveguide 2 width
period = 318e-9 # grating period
N = 1000 # number of grating periods

# simulation parameters
wr = [1530e-9, 1565e-9] # wavelength range to plot
res = 500 # number of wavelength points

# Device creation, simulation and performance assessment
device = ChirpedContraDC(w1=w1, w2=w2, N=N, period=period,
				wvl_range=wr, resolution=res)
device.simulate().displayResults()
```


<img src="figures/Example_spectrum.png" style="width: 100%">




  

    <div class="document">
      <div class="documentwrapper">
        <div class="bodywrapper">
          <div class="body" role="main">
            
  <div class="section" id="module-ContraDC">
<span id="welcome-to-contra-dc-s-documentation"></span># Welcome to Contra-DC’s documentation![¶](#module-ContraDC "Permalink to this headline")
ContraDC class
Contra-directional coupler model
Chirp your CDC, engineer your response
Based on Matlab model by Jonathan St-Yves
as well as Python model by Mustafa Hammood
Jonathan Cauchon, Created September 2019
Last updated November 2020
<dl class="class">
<dt id="ContraDC.ContraDC">
<em class="property">class </em><code class="descclassname">ContraDC.</code><code class="descname">ContraDC</code><span class="sig-paren">(</span><em>N=1000, period=3.22e-07, a=10, apod_shape='gaussian', kappa=48000, T=300, resolution=500, N_seg=100, wvl_range=[1.53e-06, 1.58e-06], central_wvl=1.55e-06, alpha=10, stages=1, w1=5.6e-07, w2=4.4e-07, w_chirp_step=1e-09, period_chirp_step=2e-09</em><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC" title="Permalink to this definition">¶</a></dt>
<dd><p>Contra-directional coupler class
Implements parameters for simulation purposes.</p>
<dl class="method">
<dt id="ContraDC.ContraDC.displayResults">
<code class="descname">displayResults</code><span class="sig-paren">(</span><em>advanced=False</em>, <em>tag_url=False</em><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.displayResults" title="Permalink to this definition">¶</a></dt>
<dd><p>Displays the result of the simulation in a user-friendly way.
Convenient for design and optimization. Interface show the device’s
specifications and grating profiles, a graph of the spectral response, 
as well as key performance figures calculated in getPerormance()</p>
</dd></dl>

<dl class="method">
<dt id="ContraDC.ContraDC.getApodProfile">
<code class="descname">getApodProfile</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.getApodProfile" title="Permalink to this definition">¶</a></dt>
<dd><p>Calculates the apodization profile,
based on the apod_profile (either “gaussian” of “tanh”)</p>
</dd></dl>

<dl class="method">
<dt id="ContraDC.ContraDC.getChirpProfile">
<code class="descname">getChirpProfile</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.getChirpProfile" title="Permalink to this definition">¶</a></dt>
<dd><p>Creates linear chirp profiles along the CDC device.
Chirp is specified by assigning 2-element lists to the constructor
for period, w1, w2 and T. The profiles are then created as linear, 
and granularity is brought by the chirp_resolution specicfications 
to match the fabrication process capabilities for realism (for instance, 
w_chirp_step is set to 1 nm because GDS has a grid resolution of 1 nm for
submission at ANT and AMF).</p>
</dd></dl>

<dl class="method">
<dt id="ContraDC.ContraDC.getGroupDelay">
<code class="descname">getGroupDelay</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.getGroupDelay" title="Permalink to this definition">¶</a></dt>
<dd><p>Calculates the groupe delay of the device,
using the phase derivative.</p>
</dd></dl>

<dl class="method">
<dt id="ContraDC.ContraDC.getPerformance">
<code class="descname">getPerformance</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.getPerformance" title="Permalink to this definition">¶</a></dt>
<dd><p>Calculated a couple of basic performance figures of the contra-DC,
such as center wavelength, bandwidth, maximum reflection, etc.</p>
</dd></dl>

<dl class="method">
<dt id="ContraDC.ContraDC.getPropConstants">
<code class="descname">getPropConstants</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.getPropConstants" title="Permalink to this definition">¶</a></dt>
<dd><p>Calculates propagation constants,
either through interpolation (for silicon), or through regression,
given a text file containing the polyfit parameters (for nitride).</p>
</dd></dl>

<dl class="method">
<dt id="ContraDC.ContraDC.makeRightShape">
<code class="descname">makeRightShape</code><span class="sig-paren">(</span><em>param</em><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.makeRightShape" title="Permalink to this definition">¶</a></dt>
<dd><p>Simply adds dimensionality to the pârameters in sights of 
matrix operations in the “propagate” method.</p>
</dd></dl>

<dl class="method">
<dt id="ContraDC.ContraDC.propagate">
<code class="descname">propagate</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.propagate" title="Permalink to this definition">¶</a></dt>
<dd><p>Propagates the optical field through the contra-DC.</p>
<p>This method uses the transfer-matrix method in a computationally-efficient
way to calculate the total transfer matrix and extract the thru and drop 
electric field responses.</p>
</dd></dl>

<dl class="method">
<dt id="ContraDC.ContraDC.simulate">
<code class="descname">simulate</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="headerlink" href="#ContraDC.ContraDC.simulate" title="Permalink to this definition">¶</a></dt>
<dd><p>Simulates the contra-DC, in logical order as prescribed by the TMM method</p>
</dd></dl>

</dd></dl>

<div class="toctree-wrapper compound">
</div>
</div>
<div class="section" id="indices-and-tables">
<h1>Indices and tables<a class="headerlink" href="#indices-and-tables" title="Permalink to this headline">¶</a></h1>
<ul class="simple">
<li><a class="reference internal" href="genindex.html"><span class="std std-ref">Index</span></a></li>
<li><a class="reference internal" href="py-modindex.html"><span class="std std-ref">Module Index</span></a></li>
<li><a class="reference internal" href="search.html"><span class="std std-ref">Search Page</span></a></li>
</ul>
</div>


          </div>
        </div>
      </div>
      <div class="sphinxsidebar" role="navigation" aria-label="main navigation">
        <div class="sphinxsidebarwrapper">
  ### [Table Of Contents]()
  
- [Welcome to Contra-DC’s documentation!]()
- [Indices and tables](#indices-and-tables)

<div class="relations">
<h3>Related Topics</h3>
<ul>
  <li><a href="">Documentation overview</a><ul>
  </ul></li>
</ul>
</div>
  <div role="note" aria-label="source link">
    ### This Page
    
      - [Show Source](_sources/index.rst.txt)
    
   </div>
<div id="searchbox" style="display: none;" role="search">
  <h3>Quick search</h3>
    <div class="searchformwrapper">
    <form class="search" action="search.html" method="get">
      <input type="text" name="q">
      <input type="submit" value="Go">
      <input type="hidden" name="check_keywords" value="yes">
      <input type="hidden" name="area" value="default">
    </form>
    </div>
</div>

        </div>
      </div>
      <div class="clearer"></div>
    </div>
    <div class="footer">
      ©2020, Jonathan Cauchon.
      
      |
      Powered by [Sphinx 1.7.4](http://sphinx-doc.org/)
      &amp; [Alabaster 0.7.10](https://github.com/bitprophet/alabaster)
      
      |
      [Page source](_sources/index.rst.txt)
    </div>

    

    
  


#!/usr/bin/env python
# coding: utf-8

# In[2]:


from collections import namedtuple
import unyt as u
import numpy as np


# In[6]:


from prepic import lwfa


# In[3]:


Cetal = namedtuple(
    "Cetal",
    [
        "npe",
        "kp",
        "f_number",
        "focal_distance",
        "beam_diameter",
        "w0",
        "fwhm",
        "zR",
        "a0",
        "ɛL",
        "τL",
        "intensity",
        "power",
    ],
)


# In[4]:


param = Cetal(
    npe=1.5e18 / u.cm ** 3,
    kp=0.2304711 * 1 / u.micrometer,
    f_number=24.9912 * u.dimensionless,
    focal_distance=3.2 * u.meter,
    beam_diameter=128.045071 * u.mm,
    w0=18 * u.micrometer,
    fwhm=21.193380405278543 * u.micrometer,
    zR=1.27234502 * u.mm,
    a0=4.076967454355432 * u.dimensionless,
    ɛL=7.7 * u.joule,
    τL=40 * u.femtosecond,
    intensity=3.553314404474785e19 * u.watt / u.cm ** 2,
    power=180.84167614968285 * u.terawatt,
)


# In[5]:


def get_plasma(parameters):
    bubble_r = (2 * np.sqrt(parameters.a0) / parameters.kp)

    return lwfa.Plasma(
        n_pe=parameters.npe,
        laser=lwfa.Laser.from_a0(
            a0=parameters.a0, ɛL=parameters.ɛL, beam=lwfa.GaussianBeam(w0=parameters.w0)
        ),
        bubble_radius=bubble_r
    )


# In[6]:


cetal_plasma = get_plasma(param)


# In[32]:


cetal_plasma.laser.beam


# In[20]:


cetal_plasma.laser.E0


# In[23]:


match = lwfa.matched_laser_plasma(param.a0)


# In[26]:


match.η


# In[28]:


sim = lwfa.Simulation(cetal_plasma)


# In[31]:


sim.__dict__


# In[34]:


sim.L


# In[35]:


sim.Δx


# In[36]:


sim.Δz


# In[37]:


sim.nx


# In[38]:


sim.nz


# In[39]:


sim.npart


# In[40]:


sim.nstep


# In[ ]:





# In[ ]:





# 1. gas-jet with $L_{\text{acc}} = 3$ mm, $n_{\text{pe}} = (1-3) \times 10^{18}$ cm${}^{-3}$
# 2. capillary with $L_{\text{acc}} = (3-10)$ cm, $n_{\text{pe}} = (3-7) \times 10^{17}$ cm${}^{-3}$

# - $w_0 = 2 \sigma_{\text{rms}}$, and experiments say $\sigma_{\text{rms}} = 7$ $\mu$m, so $w_0 = 14$ $\mu$m
# - $\varepsilon_L = 3$ J, $\tau_L = 30$ fs, $I_0 = 3 \times 10^{19}$ W/cm${}^{2}$, $a_0=3.4$

# In[3]:


npe_jet = 3e18 / u.cm ** 3
l_acc_jet = 3 * u.mm
#
npe_capil = 3e17 / u.cm ** 3
l_acc_capil = 5 * u.cm


# In[7]:


beam_frasc = lwfa.GaussianBeam(w0=15.56 * u.micrometer)
laser_frasc = lwfa.Laser(ɛL=3.0 * u.joule, τL=30 * u.femtosecond, beam=beam_frasc)


# In[17]:


plasma_jet = lwfa.Plasma(n_pe=npe_jet, laser=laser_frasc, propagation_distance=l_acc_jet)
print(plasma_jet)


# In[16]:


plasma_capil = lwfa.Plasma(
    n_pe=npe_capil, laser=laser_frasc, propagation_distance=l_acc_capil
)
print(plasma_capil)


# In[11]:


lwfa.Simulation(plasma_jet)


# In[12]:


lwfa.Simulation(plasma_capil)


# In[15]:


matched_frasc = lwfa.matched_laser_plasma(a0=3.4 * u.dimensionless)
print(matched_frasc)


# In[ ]:





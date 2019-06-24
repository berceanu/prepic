import unyt as u

from prepic import lwfa

laser_plasma = lwfa.matched_laser_plasma(a0=4.4 * u.dimensionless)
print(laser_plasma)

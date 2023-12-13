from fuel_tank import FuelTank

n_load = 8.5
n_volume = 1.2
n_safety = 1.5
M_p = 504
M_dry_0 = 742.1
M_tank_0 = 43.1
rho_p = 3100
H = 2

pmin = 20e5
pmax = 100e5

Rmin = 0.1
Rmax = 0.5

materials = {"Al2024-T3": (2780, 345e6, 73.1e9, 0.3),
             "Ti6a1-4V": (4430, 880e6, 113.8e9, 0.3)
             }

material_tank = materials["Al2024-T3"]
material_struct = materials["Al2024-T3"]

M_0 = M_dry_0 - M_tank_0   # mass of the spacecraft excluding the structure and the fuel
p = 20e5
R = 0.2

tank = FuelTank(n_load, n_volume, n_safety, H, p, M_p, rho_p, material_tank, material_struct, M_0, R)

print(tank.mass())
print(tank.t1)
print(tank.t2)
print(tank.t3)
print(tank.column_buckling_tank())
print(tank.column_bucking_struct())

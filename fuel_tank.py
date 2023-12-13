import numpy as np
import scipy as sp

g = 9.80665


class FuelTank:

    def __init__(self, n_load, n_volume, n_safety, H, p, M_p, rho_p,
                 material_tank, material_struct, M_0, R):

        self.n_load = n_load  # load factor for launch

        self.p = p * n_safety  # fuel pressure

        self.M_p = M_p  # fuel mass
        self.V = n_volume * M_p / rho_p  # fuel volume

        self.M_0 = M_0  # mass of spacecraft exlcuding fuel tank, structure and fuel itself

        self.H = H  # total height of spacecraft
        self.R = R  # radius of cylinder(s)

        self.l1 = (self.V - 4 / 3 * self.R ** 3) / (np.pi * self.R ** 2)  # length of upper cylindrical structure
        self.l2 = self.H - self.l1 - self.R  # length of cylindrical part of fuel tank

        if self.l1 < 0 or self.l2 < 0:
            raise ValueError(":3")   # raise error if things don't fit

        self.rho_tank = material_tank[0]
        self.sigma_tank = material_tank[1]
        self.E_tank = material_tank[2]
        self.nu_tank = material_tank[3]

        self.rho_struct = material_struct[0]
        self.sigma_struct = material_struct[1]
        self.E_struct = material_struct[2]
        self.nu_struct = material_struct[3]

        t3_buckling = self.get_t3_buckling()  # iterate for t3 for thin shell buckling
        self.t3 = t3_buckling
        self.M_struct = self.struct_mass(self.t3)

        t1_pressure = self.get_t1_pressure()  # get t1 for pressure
        t2_pressure = self.get_t2_pressure()  # get t2 for pressure
        t1_buckling = self.get_t1_buckling()  # iterate for t1 for thin shell buckling
        self.t2 = t2_pressure  # t2 defined by pressure
        self.t1 = np.max([t1_pressure, t1_buckling])  # t1 defined by either pressure or thin shell buckling

        self.M_tank_cylinder = self.tank_cylinder_mass(self.t1)  # get mass of cylindrical part of fuel tank
        self.M_tank_bulkheads = self.tank_cylinder_mass(self.t2)  # get combined mass of bulkheads

        self.A1 = 2 * np.pi * self.R * self.t1  # cross-sectional area of fuel tank
        self.A2 = 2 * np.pi * self.R * self.t3  # cross-sectional area of upper structure
        self.I1 = np.pi * self.R ** 3 * self.t1  # intertia of fuel tank
        self.I2 = np.pi * self.R ** 3 * self.t3  # intertia of upper structure

    def get_t1_pressure(self):
        t1_pressure = self.p * self.R / self.sigma_tank
        return t1_pressure

    def get_t2_pressure(self):
        t2_pressure = self.p * self.R / (2 * self.sigma_tank)
        return t2_pressure

    def get_t1_buckling(self):
        t1_init = 0.001  # thickness to start iteration
        t1_temp_old = t1_init

        while True:

            A_temp = 2 * np.pi * self.R * t1_temp_old   # cross-sectional area
            M_tot_temp = self.M_0 + self.M_struct + self.tank_cylinder_mass(t1_temp_old) + self.tank_bulkhead_mass(
                t1_temp_old) + self.M_p   # mass acting on fuel tank is mass of spacecraft, upper structure, its own mass and the propellant mass

            sigma = M_tot_temp * g * self.n_load / A_temp
            t1_temp_new = sp.optimize.root_scalar(lambda t: self.shell_buckling_tank(t, sigma), x0=0.001).root   # get new thickness

            if np.abs(t1_temp_old - t1_temp_new) < 0.00001:
                break
            else:
                t1_temp_old = t1_temp_new   # old thickness is new thickness
                continue

        return t1_temp_new

    def get_t3_buckling(self):
        t3_init = 0.001  # thickness to start iteration
        t3_temp_old = t3_init

        while True:

            A_temp = 2 * np.pi * self.R * t3_temp_old  # cross-sectional area
            M_tot_temp = self.M_0 + self.struct_mass(t3_temp_old)   # mass acting on upper structure is just its own mass plus the mass of the spaceraft

            sigma = M_tot_temp * g * self.n_load / A_temp
            t3_temp_new = sp.optimize.root_scalar(lambda t: self.shell_buckling_struct(t, sigma), x0=0.001).root   # get new thickness

            if np.abs(t3_temp_old - t3_temp_new) < 0.00001:
                break
            else:
                t3_temp_old = t3_temp_new   # old thickness is new thickness
                continue

        return t3_temp_new

    def shell_buckling_tank(self, t1_temp, sigma):
        lambd = np.sqrt(
            12 / np.pi ** 4 * self.l1 ** 4 / (self.R ** 2 * t1_temp ** 2) * (1 - self.nu_tank ** 2))   # optimum lambda
        Q = self.p / self.E_tank * (self.R / t1_temp) ** 2
        k = lambd + 12 / np.pi ** 4 * self.l1 ** 4 / (self.R ** 2 * t1_temp ** 2) * \
            (1 - self.nu_tank ** 2) * 1 / lambd

        sigma_cr = (1.983 - 0.983 * np.e ** (-23.14 * Q)) * k * np.pi ** 2 * \
            self.E_tank / (12 * (1 - self.nu_tank ** 2)) * (t1_temp / self.l1) ** 2
        return sigma_cr - sigma

    def shell_buckling_struct(self, t3_temp, sigma):
        lambd = np.sqrt(
            12 / np.pi ** 4 * self.l2 ** 4 / (self.R ** 2 * t3_temp ** 2) * (1 - self.nu_struct ** 2))   # optimum lambda
        Q = self.p / self.E_struct * (self.R / t3_temp) ** 2
        k = lambd + 12 / np.pi ** 4 * self.l2 ** 4 / (self.R ** 2 * t3_temp ** 2) * \
            (1 - self.nu_struct ** 2) * 1 / lambd

        sigma_cr = (1.983 - 0.983 * np.e ** (-23.14 * Q)) * k * np.pi ** 2 * \
            self.E_struct / (12 * (1 - self.nu_struct ** 2)) * (t3_temp / self.l2) ** 2
        return sigma_cr - sigma

    def tank_bulkhead_mass(self, t):
        return 4 * np.pi * self.R ** 2 * t * self.rho_tank   # returns total mass of the two bulkheads

    def tank_cylinder_mass(self, t):
        return 2 * np.pi * self.R * t * self.l2 * self.rho_tank

    def struct_mass(self, t):
        return 2 * np.pi * self.R * t * self.l2 * self.rho_struct

    def column_buckling_tank(self):
        sigma = (self.M_0 + self.M_struct) * self.n_load * g
        sigma_cr = np.pi ** 2 * self.E_tank * self.I1 / (self.A1 * self.l1 ** 2) / self.A1
        return sigma_cr - sigma

    def column_bucking_struct(self):
        sigma = (self.M_0 + self.M_struct + self.M_tank_cylinder + self.M_tank_bulkheads + self.M_p) * self.n_load * g / self.A2
        sigma_cr = np.pi ** 2 * self.E_struct * self.I2 / (self.A2 * self.l2 ** 2)
        return sigma_cr - sigma

    def mass(self):
        return self.M_tank_cylinder + self.M_tank_bulkheads + self.M_struct

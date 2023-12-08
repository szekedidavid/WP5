import numpy as np
import scipy as sp

g = 9.80665


class FuelTank:

    def __init__(self, n_load, p, M_p, rho_p, n_volume, M_0, n_safety, R, t3,
                 material_tank, material_struct, H):

        self.p = p

        self.p = self.p * n_safety

        self.M_p = M_p
        self.V = n_volume * M_p / rho_p

        self.M_0 = M_0

        self.H = H
        self.R = R

        self.l1 = (self.V - 4 / 3 * self.R ** 3) / (np.pi * self.R ** 2)
        self.l2 = self.H - self.l1

        if self.l1 < 0 or self.l2 < 0:
            raise ValueError(":3")

        self.L = self.l1 + 2 * R

        self.material_tank = material_tank
        self.material_struct = material_struct

        self.t1 = self.get_t1()
        self.t2 = self.get_t2()
        self.t3 = self.get_t3()

        self.A1 = 2 * np.pi * self.R * self.t1
        self.A2 = 2 * np.pi * self.R * self.t3
        self.I1 = np.pi * self.R ** 3 * self.t1
        self.I2 = np.pi * self.R ** 3 * self.t3

        self.sigma = M_0 * n_load * g / self.A1

    def get_t1(self):
        t1_pressure = self.p * self.R / self.material_tank.sigma
        t1_buckling = sp.optimize.root_scalar(self.shell_buckling_tank).root
        return max(t1_pressure, t1_buckling)

    def get_t2(self):
        t2_pressure = self.p * self.R / (2 * self.material_tank.sigma)
        return t2_pressure

    def get_t3(self):
        t3_buckling = sp.optimize.root_scalar(self.shell_buckling_struct).root
        return t3_buckling

    def shell_buckling_tank(self, t1_temp):
        lambd = np.sqrt(
            12 / np.pi ** 4 * self.l1 ** 4 / (self.R ** 2 * t1_temp ** 2) * (1 - self.material_tank.nu ** 2))
        Q = self.p / self.material_tank.E * (self.R / t1_temp) ** 2
        k = lambd + 12 / np.pi ** 4 * self.l1 ** 4 / (self.R ** 2 * t1_temp ** 2) * \
            (1 - self.material_tank.nu ** 2) * 1 / lambd

        sigma_cr = (1.983 - 0.983 * np.e ** (-23.14 * Q)) * k * np.pi ** 2 * \
            self.material_tank.E / (12 * (1 - self.material_tank.nu ** 2)) * (t1_temp / self.l1) ** 2
        return sigma_cr - self.sigma

    def shell_buckling_struct(self, t3_temp):
        lambd = np.sqrt(
            12 / np.pi ** 4 * self.l2 ** 4 / (self.R ** 2 * t3_temp ** 2) * (1 - self.material_struct.nu ** 2))
        Q = self.p / self.material_struct.E * (self.R / t3_temp) ** 2
        k = lambd + 12 / np.pi ** 4 * self.l2 ** 4 / (self.R ** 2 * t3_temp ** 2) * \
            (1 - self.material_struct.nu ** 2) * 1 / lambd

        sigma_cr = (1.983 - 0.983 * np.e ** (-23.14 * Q)) * k * np.pi ** 2 * \
            self.material_struct.E / (12 * (1 - self.material_struct.nu ** 2)) * (t3_temp / self.l2) ** 2
        return sigma_cr - self.sigma

    def column_buckling_tank(self, t1_temp):
        sigma_cr = np.pi ** 2 * self.material_tank.E * self.I1 / (self.A1 * self.l1 ** 2)
        return sigma_cr - self.sigma

    def column_bucking_struct(self):
        sigma_cr = np.pi ** 2 * self.material_struct.E * self.I2 / (self.A2 * self.l2 ** 2)
        return sigma_cr - self.sigma

    @property
    def mass(self):
        return self.material_tank.rho * (2 * np.pi * self.R * self.l1 * self.t1 + 4 * np.pi * self.R ** 2 * self.t2) +\
            self.material_struct.rho * 2 * np.pi * self.R * self.l2 * self.t3

import numpy as np
from material import Material


class FuelTank:

    def __init__(self, p, pmin, pmax, M_p, rho_p, n_volume, n_safety, R, material):

        self.p = p
        self.pmin = pmin
        self.pmax = pmax

        self.p = self.p * n_safety

        self.M_p = M_p
        self.V = n_volume * M_p / rho_p

        self.R = R
        self.l = self.get_l()
        self.L = self.l + 2 * R

        self.material = material

        self.t1 = self.get_t1()
        self.t2 = self.get_t2()

        self.A = 2 * np.pi * self.R * self.t1
        self.I = np.pi * self.R ** 3 * self.t1

    def check_p(self):
        if not (self.pmin < self.p < self.pmax):
            raise ValueError(":3")

    def get_l(self):
        l = (self.V - 4 / 3 * self.R ** 3) / (np.pi * self.R ** 2)
        if l < 0:
            raise ValueError(":3")
        return l

    def get_t1(self):
        return self.p * self.R / self.material.sigma

    def get_t2(self):
        return self.p * self.R / (2 * self.material.sigma)

    @property
    def column_buckling(self):
        sigma_cr = np.pi ** 2 * self.material.E * self.I / (self.A * self.l ** 2)
        return self.material.sigma - sigma_cr

    @property
    def shell_buckling(self):
        lambd = np.sqrt(12 / np.pi ** 4 * self.l ** 4 / (self.R ** 2 * self.t1 ** 2) * (4 - self.material.nu ** 2))
        Q = self.p / self.material.E * (self.R / self.t1) ** 2
        k = lambd + 12 / np.pi ** 4 * self.l ** 4 / (self.R ** 2 * self.t1 ** 2) * \
            (1 - self.material.mu ** 2) * 1 / lambd

        sigma_cr = (1.983 - 0.983 * np.e ** (-23.14 * Q)) * k * np.pi ** 2 * \
            self.material.E / (12 * (1 - self.material.mu ** 2)) * (self.t1 / self.L) ** 2
        return self.material.sigma - sigma_cr

    @property
    def mass(self):
        return self.material.rho * (2 * np.pi * self.R * self.l * self.t1 + 4 * np.pi * self.R ** 2 * self.t2)

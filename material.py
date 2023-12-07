import numpy as np


class Material:

    def __init__(self, rho, E, sigma, nu):
        self.rho = rho
        self.E = E
        self.sigma = sigma
        self.nu = nu

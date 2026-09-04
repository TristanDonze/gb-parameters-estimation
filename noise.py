# Noise curve taken from LDC used for Sangria and Spritz, "MRDv1"

from abc import ABC
import numpy as np


CLIGHT = 299792458.0
arm_length = 2.5e9


class AnalyticNoise(ABC):
    """Analytic approximation of the two components of LISA noise:
    acceleration noise and optical metrology system (OMS) noise
    """

    def __init__(self, frq, noise_type):
        """Set two components noise contributions wrt given model."""
        super().__init__()

        if noise_type == "MRDv1":
            self.DSoms_d = (10.0e-12) ** 2  # m^2/Hz
            self.DSa_a = (2.4e-15) ** 2  # m^2/sec^4/Hz

        elif noise_type == "sangria":
            self.DSoms_d = (7.9e-12) ** 2  # m^2/Hz
            self.DSa_a = (2.4e-15) ** 2  # m^2/sec^4/Hz

        else:
            raise NotImplementedError(
                "Noise type " + noise_type + " not implemented"
            )

        self.freq = np.asarray(frq)

        # Acceleration noise
        Sa_a = (
            self.DSa_a
            * (1.0 + (0.4e-3 / self.freq) ** 2)
            * (1.0 + (self.freq / 8e-3) ** 4)
        )

        self.Sa_d = Sa_a * (2.0 * np.pi * self.freq) ** (-4.0)

        Sa_nu = (
            self.Sa_d
            * (2.0 * np.pi * self.freq / CLIGHT) ** 2
        )

        self.Spm = Sa_nu

        # Optical Metrology System
        self.Soms_d = (
            self.DSoms_d
            * (1.0 + (2.0e-3 / self.freq) ** 4)
        )

        Soms_nu = (
            self.Soms_d
            * (2.0 * np.pi * self.freq / CLIGHT) ** 2
        )

        self.Sop = Soms_nu

    def psd(self, option="A", tdi2=False):
        """Return noise PSD at given freq. or freq. range.

        Option can be X, A, E, T.
        """

        lisaLT = arm_length / CLIGHT

        x = 2.0 * np.pi * lisaLT * self.freq

        if option == "X":
            S = (
                16.0
                * np.sin(x) ** 2
                * (
                    2.0 * (1.0 + np.cos(x) ** 2) * self.Spm
                    + self.Sop
                )
            )

        elif option in ["A", "E"]:
            S = (
                8.0
                * np.sin(x) ** 2
                * (
                    2.0
                    * self.Spm
                    * (
                        3.0
                        + 2.0 * np.cos(x)
                        + np.cos(2.0 * x)
                    )
                    + self.Sop * (2.0 + np.cos(x))
                )
            )

        elif option == "T":
            S = (
                16.0
                * self.Sop
                * (1.0 - np.cos(x))
                * np.sin(x) ** 2
                + 128.0
                * self.Spm
                * np.sin(x) ** 2
                * np.sin(0.5 * x) ** 4
            )

        else:
            print(
                "PSD option should be in [X, A, E, T] (%s)"
                % option
            )
            return None

        if tdi2:
            factor_tdi2 = 4 * np.sin(2 * x) ** 2
            S *= factor_tdi2

        return S
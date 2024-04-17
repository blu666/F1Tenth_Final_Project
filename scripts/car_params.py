from dataclasses import dataclass

@dataclass
class CarParams:
    wheelbase: float
    friction_coeff: float
    h_cg: float
    l_f: float
    l_r: float
    cs_f: float
    cs_r: float
    mass: float
    I_z: float
    
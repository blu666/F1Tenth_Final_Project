import numpy as np
from dataclasses import dataclass, field
from utils.params import MPCParams, initMPCParams, initLMPCParams

    

class MPC():
    def __init__(self, mpcParams: MPCParams) -> None:
        self.N      = mpcParams.N
        self.Qslack = mpcParams.Qslack
        self.Q      = mpcParams.Q
        self.Qf     = mpcParams.Qf
        self.R      = mpcParams.R
        self.dR     = mpcParams.dR
        self.n      = mpcParams.n
        self.d      = mpcParams.d
        self.A      = mpcParams.A
        self.B      = mpcParams.B
        self.Fx     = mpcParams.Fx
        self.Fu     = mpcParams.Fu
        self.bx     = mpcParams.bx
        self.bu     = mpcParams.bu
        self.xRef   = mpcParams.xRef

        self.slacks          = mpcParams.slacks
        self.timeVarying     = mpcParams.timeVarying
        
        self.predictiveModel = ... # TODO: implement predictive model

    def 

class LMPC():
    def __init__(self) -> None:
        pass


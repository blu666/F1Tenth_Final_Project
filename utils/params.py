import numpy as np
from dataclasses import dataclass, field
from utils.track import Track


#==== Vehicle Parameters
@dataclass
class CarParams:
    r_accel: float
    r_steer: float
    q_s: float
    q_s_terminal: float
    N: int
    Ts: float
    K_NEAR: int
    VEL_MAX: float
    STEER_MAX: float
    ACCELERATION_MAX: float
    DECELERATION_MAX: float
    DYNA_VEL_THRESH: float
    MAP_MARGIN: float
    WAYPOINT_SPACE: float
    wheelbase: float
    friction_coeff: float
    h_cg: float
    l_r: float
    l_f: float
    cs_f: float
    cs_r: float
    mass: float
    I_z: float
    random: float


def load_default_car_params():
    car = CarParams(r_accel=1.5,
                     r_steer=18.0,
                     q_s=3000.0,
                     q_s_terminal=800,
                     N=25,
                     Ts=0.05,
                     K_NEAR=16,
                     VEL_MAX=10.00,
                     STEER_MAX=0.41,
                     ACCELERATION_MAX=4.0,
                     DECELERATION_MAX=5.0,
                     DYNA_VEL_THRESH=0.8,
                     MAP_MARGIN=0.32,
                     WAYPOINT_SPACE=0.05,
                     wheelbase=0.3302,
                     friction_coeff=1.2,
                     h_cg=0.08255,
                     l_r=0.17145,
                     l_f=0.15875,
                     cs_f=2.3,
                     cs_r=2.3,
                     mass=3.17,
                     I_z=0.0398378,
                     random=1.0)
    return car

#==== Controller Parameters
@dataclass
class MPCParams:
    n: int = field(default=None) # state dim [vx, vy, wz, epsi, s, ey]
    d: int = field(default=None) # control dim [steer, accel]
    N: int = field(default=None) # horizon dim
    
    A: np.ndarray = field(default=None) #
    B: np.ndarray = field(default=None) #
    
    Q: np.ndarray = field(default=None) # Quad state cost
    R: np.ndarray = field(default=None) # Quad control cost
    Qf: np.ndarray = field(default=None) # Terminal state cost
    dR: np.ndarray = field(default=None) # Quad control rate cost
    
    Qslack: np.ndarray = field(default=None) # Quad slack cost # TODO: what is slack cost?
    Fx: np.ndarray = field(default=None) # STate constraint Fx * x <= bx
    bx: np.ndarray = field(default=None)
    Fu: np.ndarray = field(default=None) # Control constraint Fu * u <= bu
    bu: np.ndarray = field(default=None)
    xRef: np.ndarray = field(default=None) # Reference state
    
    slacks: bool = field(default=False) # Use slack variable
    timeVarying: bool = field(default=False) # Time varying model
    
    def __post_init__(self):
        if self.Qf is None: self.Qf = np.zeros((self.n, self.n))
        if self.dR is None: self.dR = np.zeros(self.d)
        if self.xRef is None: self.xRef = np.zeros(self.n)
    

def initMPCParams(n, d, N, v_target, carParams:CarParams):
    #==== Control Constraints
    Fx = np.array([[0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, -1]], dtype=float)
    bx = np.array([[2.],
                   [2.]]) # Fx <= bx, constraint on e_y
    
    Fu = np.array([[1, 0],
                   [-1, 0],
                   [0, 1],
                   [0, -1]], dtype=float)
    bu = np.array([[carParams.STEER_MAX],
                   [carParams.STEER_MAX],
                   [carParams.ACCELERATION_MAX],
                   [carParams.DECELERATION_MAX]], dtype=float)
    
    #==== MPC Parameters
    Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) # vx, vy, wz, epsi, s, ey
    R = np.diag([1.0, 10.0]) # delta, accel
    xRef = np.array([v_target, 0.0, 0.0, 0.0, 0.0, 0.0])
    Qslack = np.array([0.0, 50.0])
    
    mpcParameters    = MPCParams(n=n, d=d, N=N, Q=Q, R=R, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True, Qslack=Qslack)
    mpcParametersLTV = MPCParams(n=n, d=d, N=N, Q=Q, R=R, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True, Qslack=Qslack)
    return mpcParameters, mpcParametersLTV
    
def initLMPCParams(track: Track, N:int, carParams:CarParams):
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])
    bx = np.array([[0.7],   # max ey
                   [0.7]]), # max ey # TODO: non-uniform width for bx

    Fu = np.array([[1, 0],
                   [-1, 0],
                   [0, 1],
                   [0, -1]], dtype=float)
    bu = np.array([[carParams.STEER_MAX],
                   [carParams.STEER_MAX],
                   [carParams.ACCELERATION_MAX],
                   [carParams.DECELERATION_MAX]], dtype=float)

    # Safe Set Parameters
    K_NEAR = 12
    numSS_it = 4                  # Number of trajectories used at each iteration to build the safe set
    numSS_Points = K_NEAR * numSS_it    # Number of points to select from each trajectory to build the safe set

    Laps       = 40 + numSS_it      # Total LMPC laps
    TimeLMPC   = 400              # Simulation time ## TODO: WHAT

    # Tuning Parameters
    QterminalSlack  = 500 * np.diag([1, 1, 1, 1, 1, 1])  # Cost on the slack variable for the terminal constraint
    Qslack  =  1 * np.array([5, 25])                           # Quadratic and linear slack lane cost
    Q_LMPC  =  0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])     # State cost x = [vx, vy, wz, epsi, s, ey]
    R_LMPC  =  0 * np.diag([1.0, 1.0])                         # Input cost u = [delta, a]
    dR_LMPC =  5 * np.array([1.0, 10.0])                       # Input rate cost u
    n       = Q_LMPC.shape[0]
    d       = R_LMPC.shape[0]

    lmpcParameters    = MPCParams(n=n, d=d, N=N, Q=Q_LMPC, R=R_LMPC, dR=dR_LMPC, Fx=Fx, bx=bx, Fu=Fu, bu=bu, slacks=True, Qslack=Qslack)
    return numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters
    
    
    
    
    



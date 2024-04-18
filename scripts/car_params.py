from dataclasses import dataclass

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
    height_cg: float
    l_cg2rear: float
    l_cg2front: float
    C_S_front: float
    C_S_rear: float
    mass: float
    moment_inertia: float
    random: float


def load_default_car_params():
    car = CarParams()

    car.r_accel = 1.5
    car.r_steer = 18.0
    car.q_s = 3000.0
    car.q_s_terminal = 800

    car.N = 25
    car.Ts = 0.05
    car.K_NEAR = 16
    car.VEL_MAX = 10.00
    car.STEER_MAX = 0.41
    car.ACCELERATION_MAX = 4.0
    car.DECELERATION_MAX = 5.0
    car.DYNA_VEL_THRESH = 0.8

    car.MAP_MARGIN = 0.32

    car.WAYPOINT_SPACE = 0.2

    car.wheelbase = 0.3302 # meters
    car.friction_coeff = 1.2 # - (complete estimate)
    car.height_cg = 0.08255 # m (roughly measured to be 3.25 in)
    car.l_cg2rear = 0.17145 # m (decently measured to be 6.75 in)
    car.l_cg2front = 0.15875 # m (decently measured to be 6.25 in)
    car.C_S_front = 2.3 #.79 # 1/rad ? (estimated weight/4)
    car.C_S_rear = 2.3 #.79 # 1/rad ? (estimated weight/4)
    car.mass = 3.17 # kg (measured on car 'lidart')
    car.moment_inertia = .0398378 # kg m^2 (estimated as a rectangle with width and height of car and evenly distributed mass, then shifted to account for center of mass location)
    car.random = 1.0

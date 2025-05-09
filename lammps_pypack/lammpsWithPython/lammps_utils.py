import numpy as np

def calc_arc_length(coord_pair: list, is_linear: bool, para_eq: list[callable], bounds: list | None = None, n_pts: int = 200) -> float:
    '''
    Numerically calculate the arc length of a curve defined by parametric equations between the provided bounds.
    It uses the numpy gradient and trapz functions, and allows the number of points (n_pts) to be specified.

    Arguments:
        - para_eq: list[callable] (1 x 3) - [x_eq(), y_eq(), z_eq()] defining x, y, and z with single parameter
        - bounds: list = [0, 1] (1 x 2) - the start and end values of the parameter to integrate between
        - n_pts: int = 200 - the number of points used to evaluate the parametric equations for numeric integration
    
    Outputs:
        - length: float - The calculated arc-length between the start and end points along the curve.
    '''
    if is_linear is False:
        if bounds is None: bounds = [0, 1]
        t = np.linspace(bounds[0], bounds[1], n_pts)
        (x, y, z) = (para_eq[0](t), para_eq[1](t), para_eq[2](t))
        length = np.trapz(np.sqrt(np.gradient(x, t)**2 + np.gradient(y, t)**2 + np.gradient(z, t)**2), t)
    else:
        length = np.linalg.norm(coord_pair[1] - coord_pair[0])
    return length

def calc_vec_angle(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    theta_rad = np.atan2(np.linalg.norm(np.cross(vec_a, vec_b)), np.dot(vec_a, vec_b))
    theta = 180 - (180/np.pi) * theta_rad
    return theta

def compare_lists(list_a: list, list_b: list) -> bool:
    '''
    Compare two lists; Return true if the lists are equal element-wise
    Runs recursively if elements of both lists are themselves lists
    '''
    lists_match = True
    if list_a is None or list_b is None:
        lists_match = False
    elif len(list_a) != len(list_b):
        lists_match = False
    else:
        for i, val_a in enumerate(list_a):
            val_b = list_b[i]
            if not type(val_a) is type(val_b):
                lists_match = False
            elif isinstance(val_a, list):
                lists_match = compare_lists(val_a, val_b)
            elif isinstance(val_a, np.ndarray):
                if not all(val_a == val_b):
                    lists_match = False
            elif isinstance(val_a, float):
                if abs(val_a - val_b) > 10 ** -10:
                    lists_match = False
            elif not val_a == val_b:
                lists_match = False
    return lists_match
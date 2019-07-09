import ctypes

libsolver = ctypes.CDLL("./libsolver.so")
libsolver.solver.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
libsolver.solver.restype = ctypes.c_double

def compute_relic_density(T_osc, theta_i, f_a):
    global libsolver
    ans = libsolver.solver(ctypes.c_double(T_osc), ctypes.c_double(theta_i), ctypes.c_double(f_a))
    return float(ans)


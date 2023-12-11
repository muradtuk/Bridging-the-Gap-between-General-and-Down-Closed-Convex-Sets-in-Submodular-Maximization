import numpy as np
import scipy as sp
from scipy.optimize import approx_fprime, linprog
from copy import deepcopy
import cvxpy as cp


def psum(y, z):
    return y + np.multiply(1 - y, z)


def createConstraints(y, K):
    return [cp.matmul(y, K[i, :-1]) + K[i, -1] <= 0 for i in range(K.shape[0])]


def obtainY(K):
    y = cp.Variable(K.shape[1] - 1)
    constraints = createConstraints(y, K)

    objective = cp.Minimize(cp.norm(y, p="inf"))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return y.value


def obtainAandB(F, K_N, K_D, z, y, o_D, i, eps):
    grad_z = approx_fprime(xk=z, f=F)
    grad_z_y = approx_fprime(xk=psum(y,z), f=F)
    c = np.multiply(grad_z_y, 1 - z)
    a = cp.Variable(y.shape[0])
    b = cp.Variable(z.shape[0])
    constraints = []
    constraints.extend(createConstraints(a, K_N))
    constraints.extend(createConstraints(b, K_D))
    constraints.extend([cp.multiply(a + b, 1 - y) <= 1,
                        cp.matmul(cp.multiply(b, 1 - z), grad_z) >= (1 - eps) ** (i - 1) * F(o_D) - F(z)])

    objective = cp.Maximize(cp.matmul(c, a + cp.multiply(b, 1 - y)))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return a.value, b.value


def FWContinousGreedyHybridforKnownO_D(F, K_N, K_D, T, eps, o_D):
    y = obtainY(K=K_N)
    sols = np.empty((int(T / eps) + 1, y.shape[0] * 2))
    sols[0] = np.hstack((y, np.zeros(y.shape)))
    for i in range(1, int(T/eps) + 1):
        a, b = obtainAandB(F, K_N, K_D, z=sols[i-1, y.shape[0]:], y=sols[i-1, :y.shape[0]], o_D=o_D, i=i, eps=eps)
        sols[i, :y.shape[0]] = (1 - eps) * sols[i-1, :y.shape[0]] + eps * a
        sols[i, y.shape[0]:] = sols[i-1, y.shape[0]:] + eps * np.multiply(1 - sols[i-1, y.shape[0]:], b)

    sols_processed = np.apply_over_axes(lambda x: psum(x[:y.shape[0]], x[y.shape[0]:]), sols, axes=0)
    vals = np.apply_over_axes(F, sols_processed, axes=0)

    return sols_processed[np.argmax(vals)]


########################################################################################################################
###################################################### Unknown O_D #####################################################
def obtainAandBWhenODUnknown(F, K_N, K_D, z, y, i, eps, m, fixiate_a=False):
    grad_z = approx_fprime(xk=z, f=F)
    grad_z_y = approx_fprime(xk=psum(y, z), f=F)
    c = (1 - eps) * np.e ** (2 * eps * i) * np.multiply(grad_z_y, 1 - z)
    if not fixiate_a:
        a = cp.Variable(y.shape[0])
        b = cp.Variable(z.shape[0])
        constraints = []
        constraints.extend(createConstraints(a, K_N))
        constraints.extend(createConstraints(b, K_D))
        constraints.extend([a + b >= 0, a + b <= 1])

        objective = cp.Maximize(cp.matmul(c, a + cp.multiply(b, 1 - y)) + (1 - m) * np.e ** (eps * i) * (1 - eps * i) *
                                (cp.matmul(b, np.multiply(grad_z, 1 - z))))
    else:
        b = cp.Variable(z.shape[0])
        constraints = createConstraints(b, K_D)
        objective = cp.Maximize(cp.matmul(c, cp.multiply(b, 1 - y)))

    problem = cp.Problem(objective, constraints)
    problem.solve()
    if not fixiate_a:
        return a.value, b.value
    else:
        return y, b.value


def FWContinousGreedyHybrid(F, K_N, K_D, T, eps, t_s):
    y = obtainY(K=K_N)
    sols = np.empty((int(T / eps) + 1, y.shape[0] * 2))
    sols[0] = np.hstack((y, np.zeros(y.shape)))
    m = np.linalg.norm(y, ord=np.inf)
    for i in range(1, int(T / eps) + 1):
        a, b = obtainAandBWhenODUnknown(F, K_N, K_D, z=sols[i-1, y.shape[0]:], y=sols[i-1, :y.shape[0]],
                                        i=i, eps=eps, fixiate_a=not(i <= t_s / eps), m=m)

        sols[i, :y.shape[0]] = (1 - eps) * sols[i - 1, :y.shape[0]] + eps * a
        sols[i, y.shape[0]:] = sols[i - 1, y.shape[0]:] + eps * np.multiply(1 - sols[i - 1, y.shape[0]:], b)

    return psum(sols[-1, :y.shape[0]], sols[-1, y.shape[0]:])

########################################################################################################################
################################################# Online FW ############################################################

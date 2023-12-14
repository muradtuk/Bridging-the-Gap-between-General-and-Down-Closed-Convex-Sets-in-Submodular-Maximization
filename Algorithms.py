import numpy as np
from cvxpy import Variable
from scipy.optimize import approx_fprime
import cvxpy as cp
from tqdm import tqdm

def psum(y, z):
    """

    :param y:
    :param z:
    :return:
    """
    return z + np.multiply(1 - z, y)


def createConstraints(y, K, equality=False, second_strict_inequality=False):
    """

    :param y:
    :param K:
    :return:
    """

    if equality:
        if K.ndim > 1:
            return [cp.matmul(y, K[i, :-1]) + K[i, -1] == 0 for i in range(K.shape[0])]
        else:
            return [cp.matmul(y, K[:-1]) + K[-1] == 0]
    else:
        if not second_strict_inequality:
            if K.ndim > 1:
                return [cp.matmul(y, K[i, :-1]) + K[i, -1] <= 0 for i in range(K.shape[0])]
            else:
                return [cp.matmul(y, K[:-1]) + K[-1] <= 0]
        else:
            return [cp.matmul(y, K[i, :-1]) + K[i, -1] <= 0 if i != (K.shape[0] - 1) else
                    cp.matmul(y, K[i, :-1]) + K[i, -1] < 0 for i in range(K.shape[0])]


def obtainY(K, equality=True):
    """

    :param K:
    :return:
    """
    if K.ndim > 1:
        y = cp.Variable(K.shape[1] - 1)
    else:
        y = cp.Variable(K.shape[0] - 1)
    constraints = createConstraints(y, K, equality=equality)

    objective = cp.Minimize(cp.norm(y, p="inf"))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return y.value


def obtainAandB(F, K_N, K_D, z, y, o_D, i, eps):
    """

    :param F:
    :param K_N:
    :param K_D:
    :param z:
    :param y:
    :param o_D:
    :param i:
    :param eps:
    :return:
    """
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
    """

    :param F:
    :param K_N:
    :param K_D:
    :param T:
    :param eps:
    :param o_D:
    :return:
    """
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
def obtainAandBWhenODUnknown(F, K_N, K_D, z, y, i, eps, m, fixiate_a=False, gradF=None):
    """

    :param F:
    :param K_N:
    :param K_D:
    :param z:
    :param y:
    :param i:
    :param eps:
    :param m:
    :param fixiate_a:
    :param gradF:
    :return:
    """
    if gradF is None:
        grad_z = approx_fprime(xk=z, f=F)
        grad_z_y = approx_fprime(xk=psum(y, z), f=F)
    else:
        grad_z = gradF(z)
        grad_z_y = gradF(psum(y, z))

    c = np.e ** (2 * eps * i) * np.multiply(grad_z_y, 1 - z)
    if not fixiate_a:
        a: Variable = cp.Variable(y.shape[0])
        b: Variable = cp.Variable(z.shape[0])
        constraints = []
        constraints.extend(createConstraints(a, K_N, equality=True))
        constraints.extend(createConstraints(b, K_D))
        constraints.extend([a + b >= 0, a + b <= 1, a >= 0, b >= 0, a <= 1, b <= 1])

        objective = cp.Maximize(cp.matmul(c, a + cp.multiply(b, 1 - y)) + (1 - m) * np.e ** (eps * i) * (1 - eps * i) *
                                (cp.matmul(b, np.multiply(grad_z, 1 - z))))
    else:
        b = cp.Variable(z.shape[0])
        constraints = createConstraints(b, K_D, equality=False)
        constraints.extend([b >= 0, b <= 1])
        objective = cp.Maximize(cp.matmul(c, cp.multiply(b, 1 - y)))

    problem = cp.Problem(objective, constraints)
    problem.solve(cp.MOSEK)

    if not problem.status == 'optimal':
        raise ValueError('WTF happened!')

    if not fixiate_a:
        return a.value, b.value
    else:
        return y, b.value


def FWContinousGreedyHybrid(F, K_N, K_D, T, eps, t_s, gradF=None, dataset_name='', print_every=10):
    """

    :param F:
    :param K_N:
    :param K_D:
    :param T:
    :param eps:
    :param t_s:
    :return:
    """
    y = obtainY(K=K_N)
    sols = np.empty((int(T / eps) + 1, y.shape[0] * 2))
    sols[0] = np.hstack((y, np.zeros(y.shape)))
    m = np.linalg.norm(y, ord=np.inf)
    vals = [F(psum(sols[0, :y.shape[0]], sols[0, y.shape[0]:]))]
    for i in tqdm(range(1, int(T/eps) + 1)):
        a, b = obtainAandBWhenODUnknown(F, K_N, K_D, z=sols[i-1, y.shape[0]:], y=sols[i-1, :y.shape[0]],
                                        i=i, eps=eps, fixiate_a=not(i <= t_s / eps), m=m, gradF=gradF)

        sols[i, :y.shape[0]] = (1 - eps) * sols[i - 1, :y.shape[0]] + eps * a
        sols[i, y.shape[0]:] = sols[i - 1, y.shape[0]:] + eps * np.multiply(1 - sols[i - 1, y.shape[0]:], b)
        vals.append(F(psum(sols[i, :y.shape[0]], sols[i, y.shape[0]:])))
        np.save(f'data/{dataset_name}/vals.npy', vals)
        if i % print_every == 0:
            print(f'Value at iteration {i} is {vals[-1]}')

    return psum(sols[-1, :y.shape[0]], sols[-1, y.shape[0]:]), vals

########################################################################################################################
################################################# Online FW ############################################################


########################################################################################################################
################################################ Loay's Code ###########################################################
def maximizeAlongDirection(c, K):
    a = cp.Variable(c.shape[0])
    constraints = createConstraints(a, K)
    constraints.extend([a >= 0])
    objective = cp.Maximize(cp.matmul(c, a))

    problem = cp.Problem(objective, constraints)
    problem.solve(cp.MOSEK)

    if not problem.status == 'optimal':
        raise ValueError('WTF happened!')

    return a.value


def loaySolver(F, K, T, eps, gradF=None, dataset_name='', S=None, print_every=10):
    y = obtainY(K=K, equality=False)
    sols = np.empty((T, y.shape[0]))
    sols[0] = y
    vals = [F(sols[0])]
    for i in range(1, T+1):
        s = maximizeAlongDirection(K=K, c=gradF(sols[i-1]))
        sols[i] = (1 - eps) * sols[i-1] + eps * s
        vals.append(F(sols[i]))
        if i % print_every == 0:
            print(f'Value at iteration {i} is {vals[-1]}')

    np.save(f'data/{dataset_name}/loay_vals.npy', vals)

import time

import jax.numpy as jnp
from jax import jit
import numpy as np

from multiplier_method import MultiplierMethod
from gradient_descent import GradDescent


@jit
def objective(x):
    # distance from 1
    return jnp.sum((x - 1) ** 2)


@jit
def constraint(x):
    h1 = x[0] ** 2 + 0.5 * (x[1] ** 2) + x[2] ** 2 - 1
    h2 = 0.8 * (x[0] ** 2) + 2.5 * (x[1] ** 2) + x[2] ** 2 \
        + 2 * x[0] * x[2] - x[0] - x[1] - x[2] - 1
    return jnp.array([h1, h2])


def main():
    solver = MultiplierMethod(objective, constraint, num_constraint=2,
                              rho=0.1, tol=1e-4, max_iter=100)

    # optimizer to minimize augmented lagrangian.
    lagrangian_minimizer = GradDescent(
                            lr=1e-3, tol=1e-4, max_iter=100)

    # normal distribution N(1, 0).
    # np.random.seed(1)
    x = jnp.array(1 + np.random.randn(3))
    print(f'\ninitial point = {x}')

    begin = time.time()
    x_history, f_history = solver.solve(lagrangian_minimizer, x,
                                        disp=True, freq=1, history=True)

    elapsed_time = time.time() - begin

    s = 'optimal point = {}, optimal value = {}\n'
    print(s.format(x_history[-1], f_history[-1]))

    print(f'elapsed time = {elapsed_time:.2f} sec.')


if __name__ == '__main__':
    main()

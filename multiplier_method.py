import jax.numpy as jnp


class MultiplierMethod:
    """Multiplier method class."""
    def __init__(self, objective, constraint, num_constraint,
                 rho=0.1, tol=1e-4, max_iter=100):

        self.augmented_lagrangian = AugmentedLagrangian(
                                        objective, constraint,
                                        rho, jnp.ones(num_constraint))
        self.tol = tol
        self.max_iter = max_iter

    def minimize_lagragian(self, optimizer, x):
        x = optimizer.solve(self.augmented_lagrangian, x)
        return x

    def update_lambda(self, x):
        lagrangian = self.augmented_lagrangian
        lamb_new = lagrangian.lamb + lagrangian.rho * lagrangian.constraint(x)
        lagrangian.lamb = lamb_new

    def update_rho(self, hx_norm, hx_norm_prev):
        if hx_norm > 0.25 * hx_norm_prev:
            rho_new = 2 * self.augmented_lagrangian.rho
            self.augmented_lagrangian.rho = rho_new

    def is_converge(self, h_norm):
        return h_norm < self.tol

    def disp_progress(self, itr, hx_norm, fx):
        rho = self.augmented_lagrangian.rho
        print(f'iteration: {itr:4d}  |  objective: {fx:.3f}  |  h_norm: {hx_norm:.4f}  | rho: {rho:.1f}')

    def solve(self, optimizer, x_init, disp=True, freq=10, history=False):

        x = x_init
        points = [x]
        objectives = [self.augmented_lagrangian.objective(x)]
        hx_norm_prev = 1e14

        for i in range(self.max_iter):
            if disp and i % freq == 0:
                print('--------')

            x = self.minimize_lagragian(optimizer, x)
            fx = self.augmented_lagrangian.objective(x)
            points.append(x)
            objectives.append(fx)

            hx_norm = jnp.linalg.norm(self.augmented_lagrangian.constraint(x))
            if self.is_converge(hx_norm):
                self.disp_progress(i, hx_norm, fx)
                print('mulitiplier method is converged.\n')
                break

            self.update_lambda(x)
            self.update_rho(hx_norm, hx_norm_prev)
            hx_norm_prev = hx_norm

            if disp and i % freq == 0:
                self.disp_progress(i, hx_norm, fx)

        if history:
            return jnp.array(points), jnp.array(objectives)
        return points[-1], objectives[-1]


class AugmentedLagrangian():
    """Augmented lagrangian class."""
    def __init__(self, objective, constraint, rho, lamb):

        self.objective = objective
        self.constraint = constraint
        self._rho = rho
        self._lamb = lamb  # lambda

    @property
    def lamb(self):
        return self._lamb

    @lamb.setter
    def lamb(self, value):
        self._lamb = value

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = value

    def __call__(self, x):
        fx = self.objective(x)
        hx = self.constraint(x)
        penalty = 0.5 * self.rho * (jnp.linalg.norm(hx) ** 2)
        return fx + jnp.dot(self.lamb, hx) + penalty

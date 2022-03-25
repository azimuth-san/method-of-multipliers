import jax.numpy as jnp
from jax import grad, jit, jacfwd
from functools import partial


class GradDescent:
    """Gradient descent class."""
    def __init__(self, lr, tol=1e-3, max_iter=100):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter

    @partial(jit, static_argnums=(0,))
    def step(self, x, gradient):
        return x - self.lr * gradient

    @partial(jit, static_argnums=(0,))
    def is_converge(self, norm_grad):
        return norm_grad < self.tol

    def solve(self, f, x):

        if jnp.isscalar(x):
            grad_f = jit(grad(f))
        else:
            grad_f = jit(jacfwd(f))

        for i in range(self.max_iter):
            grad_fx = grad_f(x)
            norm_grad = jnp.linalg.norm(grad_fx)
            if self.is_converge(norm_grad):
                print('gradient descent is converged.')
                return x
            x = self.step(x, grad_fx)

        return x

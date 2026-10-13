"""Hessian-vector products of a neural-network training loss (Section 2.1 of the post).

The training loss of a small fully connected tanh network is
f(theta) = (1/n) sum_i (1/2) (z_i(theta) - y_i)^2. Its Hessian is applied to a
vector by differentiating the directional derivative grad f(theta)^T x once
more (Pearlmutter, 1994), and the d x d matrix is never formed, except once as
a reference for measuring errors. Requires PyTorch.
"""

from typing import Sequence

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


class TanhNetwork:
    """Fully connected network with tanh hidden layers and a linear output,
    whose parameters are stored in one flat vector theta."""

    def __init__(self, widths: Sequence[int]):
        self.shapes = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            self.shapes += [(fan_out, fan_in), (fan_out,)]
        self.size = int(sum(np.prod(shape) for shape in self.shapes))

    def initial(self, generator: torch.Generator) -> torch.Tensor:
        pieces = []
        for shape in self.shapes:
            if len(shape) == 2:
                pieces.append(torch.randn(shape, generator=generator).flatten() / np.sqrt(shape[1]))
            else:
                pieces.append(torch.zeros(shape))
        return torch.cat(pieces)

    def __call__(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        h, start = inputs, 0
        layers = len(self.shapes) // 2
        for layer in range(layers):
            rows, cols = self.shapes[2 * layer]
            weight = theta[start:start + rows * cols].view(rows, cols)
            start += rows * cols
            bias = theta[start:start + rows]
            start += rows
            h = h @ weight.T + bias
            if layer < layers - 1:
                h = torch.tanh(h)
        return h.squeeze(-1)


class TrainingLoss:
    """Squared loss of ``network`` on fixed inputs and targets."""

    def __init__(self, network: TanhNetwork, inputs: torch.Tensor, targets: torch.Tensor):
        self.network, self.inputs, self.targets = network, inputs, targets

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((self.network(theta, self.inputs) - self.targets) ** 2).mean()

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        theta = theta.detach().requires_grad_(True)
        return torch.autograd.grad(self(theta), theta)[0]

    def hvp_routine(self, theta: np.ndarray):
        """Return x -> Hessian(theta) x, computed as grad_theta(grad f(theta)^T x)."""
        point = torch.as_tensor(theta).detach().requires_grad_(True)
        gradient = torch.autograd.grad(self(point), point, create_graph=True)[0]

        def apply_h(x: np.ndarray) -> np.ndarray:
            directional = gradient @ torch.as_tensor(x)
            return torch.autograd.grad(directional, point, retain_graph=True)[0].numpy()

        return apply_h

    def dense_hessian(self, theta: np.ndarray) -> np.ndarray:
        h = torch.autograd.functional.hessian(self, torch.as_tensor(theta)).numpy()
        return 0.5 * (h + h.T)


def teacher_student_problem(widths: Sequence[int], n: int, steps: int, learning_rate: float,
                            seed: int, teacher_widths: Sequence[int] = None, n_test: int = 1024):
    """Inputs x_i ~ N(0, I), targets from a random teacher network of the same
    kind, and a student trained by full-batch gradient descent for ``steps``
    steps from a random initialization. Returns the loss and the trajectory
    (theta after 0, 1, ..., steps steps). The loss on ``n_test`` fresh inputs
    labelled by the same teacher is attached as ``loss.test``; these inputs are
    drawn after the initialization, so they do not change the training run."""
    generator = torch.Generator().manual_seed(seed)
    teacher = TanhNetwork(teacher_widths or widths)
    teacher_theta = teacher.initial(generator) * 2.0
    inputs = torch.randn((n, widths[0]), generator=generator)
    targets = teacher(teacher_theta, inputs).detach()
    student = TanhNetwork(widths)
    loss = TrainingLoss(student, inputs, targets)
    theta = student.initial(generator)
    test_inputs = torch.randn((n_test, widths[0]), generator=generator)
    loss.test = TrainingLoss(student, test_inputs, teacher(teacher_theta, test_inputs).detach())
    trajectory = [theta.numpy().copy()]
    for _ in range(steps):
        theta = theta - learning_rate * loss.gradient(theta)
        trajectory.append(theta.numpy().copy())
    return loss, trajectory

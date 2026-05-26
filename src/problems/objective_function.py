import numpy as np

class ObjectiveFunction:
    def __init__(self, dim: int , num_objectives: int , num_constraints: int):
        self.dim = dim
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.bounds = np.array([(0.0, 1.0)] * dim)  # Default bounds
        self.ref_point = None
        self.evaluator = None

    def __call__(self, x: np.ndarray):
        raise NotImplementedError("This method should be overridden by subclasses.")
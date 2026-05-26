import torch
from moo_constraints.problems.objective_function import ObjectiveFunction
from moo_constraints.problems.reference_points import ref_points
from moo_constraints.problems.reproblem import CRE21 as Source_ConTwoBarTrussDesign
from moo_constraints.problems.reproblem import CRE22 as Source_ConWeldedBeamDesign
from moo_constraints.problems.reproblem import CRE23 as Source_ConDiscBrakeDesign
from moo_constraints.problems.reproblem import CRE24 as Source_ConSpeedReducerDesign
from moo_constraints.problems.reproblem import CRE25 as Source_ConGearTrainDesign
from moo_constraints.problems.reproblem import CRE31 as Source_ConCarSideImpactDesign
from moo_constraints.problems.reproblem import CRE32 as Source_ConMarineDesign
from moo_constraints.problems.reproblem import CRE51 as Source_ConWaterResourcePlanning

import torch
import numpy as np

class BaseCREProblem(ObjectiveFunction):
    def __init__(self, base_name, core_function, **kwargs):
        self.core_function = core_function
        self.dim = core_function.n_variables
        self.num_objectives = core_function.n_objectives
        self.num_constraints = core_function.n_constraints
        
        self.lbound = torch.as_tensor(self.core_function.lbound, dtype=torch.double)
        self.ubound = torch.as_tensor(self.core_function.ubound, dtype=torch.double)
        
        self.__base_bounds = torch.stack([self.lbound, self.ubound], dim=0)

        self.bounds = torch.tensor([(0.0, 1.0)] * self.dim, dtype=torch.double).T
        
        self.ref_point = torch.tensor(ref_points.get(base_name, None), dtype=torch.double)
        self.name = f'{base_name}'

        self._last_g = None

    def _unscale_input(self, X_batch):
        return self.__base_bounds[0, :] + (self.__base_bounds[1, :] - self.__base_bounds[0, :]) * X_batch

    def evaluate_objectives(self, X_batch):
        if X_batch.ndim == 1:
            X_batch = X_batch.unsqueeze(0)
            
        X_unscaled = self._unscale_input(X_batch)
        X_numpy = X_unscaled.detach().cpu().numpy()
        
        fs = []
        gs = []
        
        for x in X_numpy:
            out = self.core_function.evaluate(x)
                
            if self.num_constraints > 0:
                f, g = out
                fs.append(f)
                gs.append(g)
            else:
                fs.append(out)
                gs.append(np.array([]))
            
        f_tensor = torch.from_numpy(np.array(fs)).to(dtype=torch.double, device=X_batch.device)
        g_tensor = torch.from_numpy(np.array(gs)).to(dtype=torch.double, device=X_batch.device)
        
        self._last_g = g_tensor
        
        return -f_tensor

    def evaluate_slack(self, X_batch):
        if self.num_constraints == 0:
            return torch.empty(X_batch.shape[0], 0, dtype=torch.double, device=X_batch.device)
        
        if self._last_g is None or self._last_g.shape[0] != X_batch.shape[0]:
            self.evaluate_objectives(X_batch)
            
        return self._last_g
    
    def __call__(self, X):
        return self.evaluate_objectives(X)
    
class BaseConTwoBarTrussDesign(BaseCREProblem):
    def __init__(self, **kwargs):
        super().__init__('con-two-bar-truss-design', Source_ConTwoBarTrussDesign(), **kwargs)

class BaseConWeldedBeamDesign(BaseCREProblem):
    def __init__(self, **kwargs):
        super().__init__('con-welded-beam-design', Source_ConWeldedBeamDesign(), **kwargs)


class BaseConDiscBrakeDesign(BaseCREProblem):
    def __init__(self, **kwargs):
        super().__init__('con-disc-brake-design', Source_ConDiscBrakeDesign(), **kwargs)


class BaseConSpeedReducerDesign(BaseCREProblem):
    def __init__(self, **kwargs):
        super().__init__('con-speed-reducer-design', Source_ConSpeedReducerDesign(), **kwargs)


class BaseConMarineDesign(BaseCREProblem):
    def __init__(self, **kwargs):
        super().__init__('con-marine-design', Source_ConMarineDesign(), **kwargs)

class BaseConGearTrainDesign(BaseCREProblem):
    def __init__(self, **kwargs):
        super().__init__('con-gear-train-design', Source_ConGearTrainDesign(), **kwargs)

class BaseConCarSideImpactDesign(BaseCREProblem):
    def __init__(self, **kwargs):
        super().__init__('con-car-side-impact-design', Source_ConCarSideImpactDesign(), **kwargs)

class BaseConWaterResourcePlanning(BaseCREProblem):
    def __init__(self, **kwargs):
        super().__init__('con-water-resource-planning', Source_ConWaterResourcePlanning(), **kwargs)



if __name__ == '__main__':

    #four-bar-truss concrete-beam pressure-vessel hatch-cover coil-spring two-bar-truss welded-beam disc-brake vehicle-design speed-reducer gear-train rocket-injector car-impact marine-design water-planning car-cab-design
    ...

"""Wrappers around unconstrained real-world multi-objective benchmark problems."""

import torch
from .objective_function import ObjectiveFunction
from .reference_points import ref_points
from .reproblem import CRE32 as Source_ConMarineDesign
from .reproblem import Knee1 as Source_Knee1
from .reproblem import Knee5 as Source_Knee5
from .reproblem import RE21 as Source_FourBarTruss
from .reproblem import RE22 as Source_ReinforcedConcreteBeam
from .reproblem import RE23 as Source_PressureVessel
from .reproblem import RE24 as Source_HatchCover
from .reproblem import RE25 as Source_CoilCompressionSpring
from .reproblem import RE31 as Source_TwoBarTruss
from .reproblem import RE32 as Source_WeldedBeam
from .reproblem import RE33 as Source_DiscBrake
from .reproblem import RE34 as Source_VehicleDesign
from .reproblem import RE35 as Source_SpeedReducer
from .reproblem import RE36 as Source_GearTrain
from .reproblem import RE37 as Source_RocketInjector
from .reproblem import RE41 as Source_CarSideImpact
from .reproblem import RE42 as Source_ConceptualMarineDesign
from .reproblem import RE61 as Source_WaterResourcePlanning
from .reproblem import RE91 as Source_CarCabDesign

class BaseREProblem(ObjectiveFunction):
    """Normalize a real-world benchmark to the common STAGE-BO interface."""
    def __init__(self, base_name, core_function, **kwargs):
        super().__init__(dim=core_function.n_variables, num_objectives=core_function.n_objectives, num_constraints=core_function.n_constraints)
        self.core_function = core_function
        self.num_constraints = 0
        self.lbound = torch.from_numpy(self.core_function.lbound).double()
        self.ubound = torch.from_numpy(self.core_function.ubound).double()
        self.__base_bounds = torch.stack(
            [
                torch.as_tensor(self.lbound, dtype=torch.double),
                torch.as_tensor(self.ubound, dtype=torch.double),
            ],
            dim=0,
        )

        self.bounds = torch.tensor([(0.0, 1.0)] * self.core_function.n_variables).T
        self.dim = self.core_function.n_variables
        self.num_objectives = self.core_function.n_objectives
        self.ref_point = torch.tensor(ref_points.get(base_name, None), dtype=torch.double)
        assert self.ref_point is not None
        self.name = f'{base_name}'

    def evaluate_objectives(self, X_batch):
        assert X_batch.ndim == 2 and X_batch.shape[1] == self.dim
        assert torch.all(X_batch <= self.bounds[1, :]) and torch.all(X_batch >= self.bounds[0, :])
        fs = []
        for X_1d in X_batch:
            f = self.evaluate_objectives_single(X_1d)
            fs.append(f)
        return torch.stack(fs)
    
    def evaluate_objectives_single(self, X_1d): 
        assert X_1d.ndim == 1 and X_1d.shape[0] == self.dim
        X_1d_scaled = self.__base_bounds[0, :] + (self.__base_bounds[1, :] - self.__base_bounds[0, :]) * X_1d
        assert torch.all(X_1d_scaled <= self.ubound) and torch.all(X_1d_scaled >= self.lbound)
        X_numpy = X_1d_scaled.detach().cpu().numpy()
        f_numpy = self.core_function.evaluate(X_numpy)
        return torch.from_numpy(f_numpy).double()
    
    def __call__(self, X):
        assert X.ndim == 2 and X.shape[1] == self.dim
        assert torch.all(X <= self.bounds[1, :]) and torch.all(X >= self.bounds[0, :])
        return -self.evaluate_objectives(X)
    
class BaseKnee1(BaseREProblem):
    def __init__(self, **kwargs):
        # d=3 n=2
        super().__init__('knee1', Source_Knee1(), **kwargs)

class BaseFourBarTruss(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 4 n=2
        super().__init__('four-bar-truss', Source_FourBarTruss(), **kwargs)

class BaseReinforcedConcreteBeam(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 3 n=2
        super().__init__('concrete-beam', Source_ReinforcedConcreteBeam(), **kwargs)

class BasePressureVessel(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 4 n=2
        super().__init__('pressure-vessel', Source_PressureVessel(), **kwargs)

class BaseHatchCover(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 2 n=2
        super().__init__('hatch-cover', Source_HatchCover(), **kwargs)

class BaseCoilCompressionSpring(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 3 n=2
        super().__init__('coil-spring', Source_CoilCompressionSpring(), **kwargs)

class BaseTwoBarTruss(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 3 n=3
        super().__init__('two-bar-truss', Source_TwoBarTruss(), **kwargs)

class BaseWeldedBeam(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 4 n=3
        super().__init__('welded-beam', Source_WeldedBeam(), **kwargs)

class BaseDiscBrake(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 4 n=3
        super().__init__('disc-brake', Source_DiscBrake(), **kwargs)

class BaseVehicleDesign(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 5 n=3
        super().__init__('vehicle-design', Source_VehicleDesign(), **kwargs)

class BaseSpeedReducer(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 7 n=3
        super().__init__('speed-reducer', Source_SpeedReducer(), **kwargs)

class BaseGearTrain(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 4 n=3
        super().__init__('gear-train', Source_GearTrain(), **kwargs)

class BaseRocketInjector(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 4 n=3
        super().__init__('rocket-injector', Source_RocketInjector(), **kwargs)


class BaseCarSideImpact(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 7 n=4
        super().__init__('car-impact', Source_CarSideImpact(), **kwargs)

class BaseConceptualMarineDesign(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 6 n=4
        super().__init__('marine-design', Source_ConceptualMarineDesign(), **kwargs)

class BaseWaterResourcePlanning(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 3 n=6
        super().__init__('water-planning', Source_WaterResourcePlanning(), **kwargs)

class BaseCarCabDesign(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 7 n=9
        super().__init__('car-cab-design', Source_CarCabDesign(), **kwargs)

class BaseKnee5(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 2 n=3
        super().__init__('knee5', Source_Knee5(), **kwargs)

class BaseCOnMarineDesign(BaseREProblem):
    def __init__(self, **kwargs):
        # d= 7 n=10
        super().__init__('con-marine-design', Source_ConMarineDesign(), **kwargs)

if __name__ == '__main__':

    #four-bar-truss concrete-beam pressure-vessel hatch-cover coil-spring two-bar-truss welded-beam disc-brake vehicle-design speed-reducer gear-train rocket-injector car-impact marine-design water-planning car-cab-design
    ...

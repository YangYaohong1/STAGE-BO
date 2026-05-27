"""Problem registry for benchmark and real-world STAGE-BO test cases."""

import numpy as np
import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.test_functions.multi_objective import (
    BNH,
    C2DTLZ2,
    CONSTR,
    CarSideImpact,
    ConstrainedBraninCurrin,
    DH4,
    DTLZ2,
    DTLZ5,
    DTLZ7,
    DiscBrake,
    BraninCurrin,
    MW7,
    OSY,
    Penicillin,
    SRN,
    ToyRobust,
    VehicleSafety,
    WeldedBeam,
    ZDT1,
    ZDT2,
    ZDT3,
)
from pymoo.problems import get_problem as pymoo_get

from problems.definitions.constrained_realworld import (
    BaseConCarSideImpactDesign,
    BaseConDiscBrakeDesign,
    BaseConGearTrainDesign,
    BaseConMarineDesign,
    BaseConSpeedReducerDesign,
    BaseConWaterResourcePlanning,
    BaseConWeldedBeamDesign,
)
from problems.definitions.realworld import (
    BaseCarCabDesign,
    BaseCarSideImpact,
    BaseCoilCompressionSpring,
    BaseConceptualMarineDesign,
    BaseDiscBrake,
    BaseFourBarTruss,
    BaseGearTrain,
    BaseHatchCover,
    BaseKnee1,
    BaseKnee5,
    BasePressureVessel,
    BaseReinforcedConcreteBeam,
    BaseRocketInjector,
    BaseSpeedReducer,
    BaseTwoBarTruss,
    BaseVehicleDesign,
    BaseWaterResourcePlanning,
    BaseWeldedBeam,
)
from runtime import tkwargs


class PymooWrapper(MultiObjectiveTestProblem):
    """Adapt a pymoo problem so it behaves like a BoTorch test problem."""
    def __init__(self, pymoo_problem_instance, negate=True):
        self.pymoo_problem = pymoo_problem_instance
        self.dim = self.pymoo_problem.n_var
        self.num_objectives = self.pymoo_problem.n_obj
        lb, ub = self.pymoo_problem.bounds()
        self._bounds = [(l, u) for l, u in zip(lb, ub)]
        super().__init__(noise_std=None, negate=negate)
        self.register_buffer("bounds", torch.tensor(np.stack([lb, ub]), dtype=torch.double))

    def _evaluate_true(self, X):
        x_flat = X.view(-1, self.dim).detach().cpu().numpy()
        out = {}
        self.pymoo_problem._evaluate(x_flat, out)
        result = torch.from_numpy(out["F"]).to(X)
        if self.negate:
            result = -result
        return result.view(X.shape[:-1] + torch.Size([self.num_objectives]))


def get_test_problem(name: str):
    """Return a configured benchmark problem by its public registry name."""
    if name == "DTLZ2_2_5":
        return DTLZ2(dim=5, num_objectives=2, negate=True).to(**tkwargs)
    if name == "DTLZ2_2_3":
        return DTLZ2(dim=3, num_objectives=2, negate=True).to(**tkwargs)
    if name == "DTLZ2_3":
        return DTLZ2(dim=4, num_objectives=3, negate=True).to(**tkwargs)
    if name == "DTLZ2_4":
        return DTLZ2(dim=5, num_objectives=4, negate=True).to(**tkwargs)
    if name == "DTLZ2_5":
        return DTLZ2(dim=6, num_objectives=5, negate=True).to(**tkwargs)
    if name == "DTLZ5_3_4":
        return DTLZ5(dim=4, num_objectives=3, negate=True).to(**tkwargs)
    if name == "DTLZ7_2":
        return DTLZ7(dim=5, num_objectives=2, negate=True).to(**tkwargs)
    if name == "DTLZ7_3":
        return DTLZ7(dim=5, num_objectives=3, negate=True).to(**tkwargs)
    if name == "DTLZ7_5":
        return DTLZ7(dim=6, num_objectives=5, negate=True).to(**tkwargs)
    if name == "DTLZ7_6":
        return DTLZ7(dim=8, num_objectives=6, negate=True).to(**tkwargs)
    if name == "ToyRobust":
        return ToyRobust(negate=False).to(**tkwargs)
    if name == "BraninCurrin":
        return BraninCurrin(negate=True).to(**tkwargs)
    if name == "ZDT1":
        return ZDT1(dim=2, num_objectives=2, negate=True).to(**tkwargs)
    if name == "ZDT1_10":
        return ZDT1(dim=10, num_objectives=2, negate=True).to(**tkwargs)
    if name == "ZDT2":
        return ZDT2(dim=2, num_objectives=2, negate=True).to(**tkwargs)
    if name == "ZDT2_10":
        return ZDT2(dim=10, num_objectives=2, negate=True).to(**tkwargs)
    if name == "ZDT2_2":
        return ZDT2(dim=2, num_objectives=2, negate=True).to(**tkwargs)
    if name == "ZDT2_8":
        return ZDT2(dim=8, num_objectives=2, negate=True).to(**tkwargs)
    if name == "ZDT3":
        return ZDT3(dim=2, num_objectives=2, negate=True).to(**tkwargs)
    if name == "ZDT3_10":
        return ZDT3(dim=10, num_objectives=2, negate=True).to(**tkwargs)
    if name == "Penicillin":
        return Penicillin(negate=True).to(**tkwargs)
    if name == "VehicleSafety":
        return VehicleSafety(negate=True).to(**tkwargs)
    if name == "CarSideImpact":
        return CarSideImpact(negate=True).to(**tkwargs)
    if name == "DH4":
        return DH4(dim=3, negate=True).to(**tkwargs)
    if name == "C2DTLZ2":
        return C2DTLZ2(dim=10, num_objectives=2, negate=True).to(**tkwargs)
    if name == "C2DTLZ2_3":
        return C2DTLZ2(dim=3, num_objectives=2, negate=True).to(**tkwargs)
    if name == "C2DTLZ2_4":
        return C2DTLZ2(dim=4, num_objectives=2, negate=True).to(**tkwargs)
    if name == "DiscBrake":
        return DiscBrake(negate=True).to(**tkwargs)
    if name == "MW7_4":
        return MW7(dim=4, negate=True).to(**tkwargs)
    if name == "OSY":
        return OSY(negate=True).to(**tkwargs)
    if name == "SRN":
        return SRN(negate=True).to(**tkwargs)
    if name == "WeldedBeam":
        return WeldedBeam(negate=True).to(**tkwargs)
    if name == "BNH":
        return BNH(negate=True).to(**tkwargs)
    if name == "CONSTR":
        return CONSTR(negate=True).to(**tkwargs)
    if name == "ConstrainedBraninCurrin":
        return ConstrainedBraninCurrin(negate=True).to(**tkwargs)
    if name == "marine-design":
        return BaseConceptualMarineDesign(**tkwargs)
    if name == "two-bar-truss":
        return BaseTwoBarTruss(**tkwargs)
    if name == "welded-beam":
        return BaseWeldedBeam(**tkwargs)
    if name == "disc-brake":
        return BaseDiscBrake(**tkwargs)
    if name == "gear-train":
        return BaseGearTrain(**tkwargs)
    if name == "rocket-injector":
        return BaseRocketInjector(**tkwargs)
    if name == "car-cab-design":
        return BaseCarCabDesign(**tkwargs)
    if name == "water-planning":
        return BaseWaterResourcePlanning(**tkwargs)
    if name == "vehicle-design":
        return BaseVehicleDesign(**tkwargs)
    if name == "speed-reducer":
        return BaseSpeedReducer(**tkwargs)
    if name == "car-impact":
        return BaseCarSideImpact(**tkwargs)
    if name == "four-bar-truss":
        return BaseFourBarTruss(**tkwargs)
    if name == "concrete-beam":
        return BaseReinforcedConcreteBeam(**tkwargs)
    if name == "pressure-vessel":
        return BasePressureVessel(**tkwargs)
    if name == "hatch-cover":
        return BaseHatchCover(**tkwargs)
    if name == "coil-spring":
        return BaseCoilCompressionSpring(**tkwargs)
    if name == "knee1":
        return BaseKnee1(**tkwargs)
    if name == "knee5":
        return BaseKnee5(**tkwargs)
    if name == "con-marine-design":
        return BaseConMarineDesign(**tkwargs)
    if name == "con-welded-beam-design":
        return BaseConWeldedBeamDesign(**tkwargs)
    if name == "con-disc-brake-design":
        return BaseConDiscBrakeDesign(**tkwargs)
    if name == "con-speed-reducer-design":
        return BaseConSpeedReducerDesign(**tkwargs)
    if name == "con-gear-train-design":
        return BaseConGearTrainDesign(**tkwargs)
    if name == "con-car-side-impact-design":
        return BaseConCarSideImpactDesign(**tkwargs)
    if name == "con-water-resource-planning":
        return BaseConWaterResourcePlanning(**tkwargs)
    try:
        return PymooWrapper(pymoo_get(name), negate=True).to(**tkwargs)
    except Exception as exc:
        raise ValueError(f"Invalid problem: {name}") from exc

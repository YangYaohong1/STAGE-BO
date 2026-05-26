import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def fit_model_list(train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    models = []
    x = train_x.to(dtype=torch.float64)
    y = train_y.to(dtype=torch.float64)
    for column in range(y.shape[-1]):
        models.append(
            SingleTaskGP(
                x,
                y[..., column : column + 1],
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    with gpytorch.settings.cholesky_jitter(1e-3):
        fit_gpytorch_mll(mll)
    return model

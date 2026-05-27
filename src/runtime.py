"""Shared runtime settings such as device and default tensor dtype."""

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {"device": DEVICE, "dtype": torch.double}

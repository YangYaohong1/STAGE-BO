import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {"device": DEVICE, "dtype": torch.double}

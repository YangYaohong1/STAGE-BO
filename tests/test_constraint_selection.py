import torch

from moo_constraints.algorithms.constraint_selection import select_constraint_via_maxmin


def test_select_constraint_via_maxmin_shapes():
    posterior = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.4],
            [0.4, 0.8],
        ],
        dtype=torch.double,
    )
    train = torch.tensor(
        [
            [0.9, 0.1],
            [0.5, 0.5],
        ],
        dtype=torch.double,
    )
    bounds, selected, _, objective_index = select_constraint_via_maxmin(posterior, train, iteration=1)
    assert bounds.shape == (1,)
    assert selected.shape == (2,)
    assert objective_index in {0, 1}

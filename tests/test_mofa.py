import numpy as np

from mosurv.mofa import project_factors


def test_projection_recovers_factors():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 3))
    Ws = [rng.normal(size=(40, 3)), rng.normal(size=(25, 3))]
    views = [Z @ W.T + 0.05 * rng.normal(size=(200, W.shape[0])) for W in Ws]
    Zh = project_factors(views, Ws, [400.0, 400.0])
    for k in range(3):
        assert np.corrcoef(Z[:, k], Zh[:, k])[0, 1] > 0.99

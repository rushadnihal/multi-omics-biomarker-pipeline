import numpy as np
import yaml
from sksurv.metrics import concordance_index_censored

import mosurv


def test_import_and_config():
    assert mosurv.__version__
    with open("config/config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["endpoint"]["primary"] == "OS"


def test_cindex_runs():
    rng = np.random.default_rng(0)
    t = rng.exponential(10, 50)
    e = rng.random(50) > 0.3
    c = concordance_index_censored(e, t, -t)[0]
    assert c > 0.9  # risk = -time is perfectly concordant

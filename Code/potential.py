from collections import namedtuple

import numpy as np

PotentialModel = namedtuple("PotentialModel", ["V", "dVdtheta"])

harmonic = PotentialModel(V=lambda theta: 0.5 * theta**2, dVdtheta=lambda theta: theta)

cosine = PotentialModel(V=lambda theta: 1 - np.cos(theta), dVdtheta=lambda theta: np.sin(theta))

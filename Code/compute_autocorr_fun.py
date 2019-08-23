import sys
import numpy as np
import emcee

input_filename = sys.argv[1]
output_filename = sys.argv[2]
data = np.load(input_filename)
samples = data["samples"].reshape((-1, 5))
fn = emcee.autocorr.function(samples)
np.save(output_filename, fn)


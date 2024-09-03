
from balltracking import balltrack as blt
import test_inputs

_ = blt.full_calibration(test_inputs.datacube_file, test_inputs.bt_params, test_inputs.cal_args, test_inputs.cal_opt_args, verbose=True)

print('success! Bravo!')

import os
import pandas as pd
import csv

os.chdir(os.path.expanduser('~/dev/sdo-tracking-framework'))
# directory hosting the drifted data (9 series)
drift_dir = os.path.join(os.environ['DATA'], 'sanity_check/stein_series/')
# output directory for the drifting images
outputdir = os.path.join(drift_dir, 'calibration')
# number of parameter sets
nsets = 5040
filelist = [os.path.join(outputdir, 'param_sweep_{:d}.csv'.format(idx)) for idx in range(nsets)]

df_list = [pd.read_csv(f) for f in filelist]
df = pd.concat(df_list, axis=0, ignore_index=True)
df.set_index('index', inplace=True)
df.to_pickle(os.path.join(drift_dir, 'calibration_dataframe.pkl'))

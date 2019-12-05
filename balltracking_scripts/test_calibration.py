import os
import pandas as pd
import numpy as np

# directory hosting the drifted data (9 series)
drift_dir = os.path.join(os.environ['DATA'], 'sanity_check/stein_series/')
# Load correlation dataframe
df = pd.read_pickle(os.path.join(drift_dir, 'correlation_dataframe.pkl'))
# Extract velocity from dataframe keys
drift_keys = [v for v in df.keys().values if 'vx_' in v]
drift_rates = [float(v.replace('vx_top ', '')) for v in drift_keys]

argmax = np.argmax(df['corr'].values)
vmeans = df[drift_keys].iloc[argmax].values

p1, r1, _, _, _ = np.polyfit(drift_rates, vmeans, 1, full=True)
print(1/p1[0])
p2, r2, _, _, _ = np.polyfit(vmeans, drift_rates, 1, full=True)
print(p2[0])
from importlib import reload
import DRMS.queries as queries
import urllib

# JSOC DRMS series name
seriesname = 'aia.lev1_euv_12s'
# Time(s) of record
t_rec = '2010.05.13_00:00'
# Wavelength(s) as strings. Put an empty string for getting all wavelengths
wavelength = '94'

# Local directory where to download the files
localpath  = '/Users/rattie'

# Build the jsoc query as an url
fits_url, filepath = queries.query_spikes(seriesname, t_rec, localpath, wavelength=wavelength)

urllib.request.urlretrieve(fits_url, filepath)

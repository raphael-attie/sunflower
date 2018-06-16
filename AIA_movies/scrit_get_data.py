import drms

c = drms.Client(email='raphael.attie@nasa.gov', verbose=True)
series = 'aia.lev1_euv_12s'
si = c.info(series)
wavelen = 171
tsel = '2012-08-31T00:00:01Z/1d@6h'
# DRMS query string
qstr = '%s[%s][%d]{image}' % (series, tsel, wavelen)

r = c.export(qstr, method='url', protocol='fits')
r.wait()
r.download('/Users/rattie/Data/SDO/AIA/')
import DRMS.queries as queries

t_start = '2011.06.27_22:00:00'
t_stop  = '2011.06.28_22:00:00'
projection = 'LambertCylindrical'
localpath = '/Users/rattie/Data/SDO/HMI/EARs/AR11242'

seriesname_list = ['su_attie.mtrack_Ic_45s_512px',
'su_attie.mtrack_M_45s_512px',
'su_attie.mtrack_V_45s_512px']

files = [queries.query_mtrack(seriesname, t_start, t_stop, projection, localpath) for seriesname in seriesname_list]

# AR 12673
t_start = '2017.09.01_00:00:00'
t_stop  = '2017.09.05_23:59:59'
projection = 'LambertCylindrical'
localpath = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01'

seriesname_list = ['su_attie.mtrack_Ic_45s_512px',
'su_attie.mtrack_M_45s_512px']

files = [queries.query_mtrack(seriesname, t_start, t_stop, projection, localpath) for seriesname in seriesname_list]

# Sometimes we just need track a single cube. But we can still have a one-element list

seriesname_list = ['su_attie.mtrack_V_45s_512px']
files = [queries.query_mtrack(seriesname, t_start, t_stop, projection, localpath) for seriesname in seriesname_list]

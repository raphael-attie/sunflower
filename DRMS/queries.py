import os
import urllib
import sys

def query_mtrack(seriesname, t_start, t_stop, projection, localpath):
    """
    Download mtracked cube from DRMS to local disk.

    :param seriesname: e.g: su_attie.mtrack_M_45s_512px
    :param t_start: start time of the fits cube. format: 2010.11.26_22:00:34_TAI
    :param t_stop: stop time. Same format as t_start
    :param projection: 'Postel' or 'LambertCylindrical'
    :param localpath: directory where to download the show_info data and fits file
    :return: download to fits file on local disk.
    """

    # Check the series name for segment information
    if 'ic_' in seriesname.lower():
        segment = 'continuum'
    elif 'v_' in seriesname.lower():
        segment =  'Dopplergram'
    elif 'm_' in seriesname.lower():
        segment = 'magnetogram'
    else:
        sys.stderr.write("error: could not recognize the seriesname")
        return None

    # Build query string into url
    JSOC2 = "http://jsoc2.stanford.edu"
    SHOWI = JSOC2 + "/cgi-bin/ajax/show_info?"
    ds = "ds=" + seriesname + "[%s][%s][][%s]"%(t_start, t_stop, projection)
    cmd = ds + "&P=1&k=1&v=1&n=-1&key=t_start,t_stop,duration"
    url = SHOWI + cmd

    # Setup basic http authentification
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, 'http://jsoc2.stanford.edu', 'hmiteam', 'hmiteam')
    auth = urllib.request.HTTPBasicAuthHandler(manager)
    opener = urllib.request.build_opener(auth)
    urllib.request.install_opener(opener)

    # Get the show_info output on a file
    show_info_file = os.path.join(localpath, 'show_info_out.txt')

    print("Getting show_info data: %s" % url)
    urllib.request.urlretrieve(url, show_info_file)

    urllib.request.urlcleanup()

    look_for = "SUDIR="
    with open(show_info_file, "r") as file_to_read:
        for line in file_to_read:
            if look_for in line:
                sudir_line= line.strip()

    print(sudir_line)
    # SUMS directory
    sums_dir = sudir_line.replace('SUDIR=', '')

    # Build the url of the fits file using the SUMS path

    filename = 'mtrack_' + t_start.replace('.','').replace(':','') + '_TAI' + \
                  t_stop.replace('.','').replace(':','') + '_' + \
                  projection + '_' + segment + '.fits'
    fits_url = JSOC2 + sums_dir + '/' + segment+ '.fits'
    filepath = os.path.join(localpath, filename)
    urllib.request.urlretrieve(fits_url, filepath)

    return filepath

def query_spikes(seriesname, t_rec, localpath, wavelength=''):
    """
    Download mtracked cube from DRMS to local disk.

    :param seriesname: e.g: su_attie.mtrack_M_45s_512px
    :param t_rec: record time of the fits file. format: 2010.11.26_22:00:34
    :param wavelength: string giving the wavelength for the record. Empty string for all
    :param localpath: directory where to download the show_info data and fits file
    :return: download to fits file on local disk.
    """

    # Build query string into url
    JSOC = "http://jsoc.stanford.edu"
    SHOWI = JSOC + "/cgi-bin/ajax/show_info?"
    ds = "ds=" + seriesname + "[%s][%s]"%(t_rec, wavelength)
    cmd = ds + "&P=1&k=1&v=1&n=-1&seg=spikes&key=t_rec,wavelnth"
    url = SHOWI + cmd

    # Get the show_info output on a file
    show_info_file = os.path.join(localpath, 'show_info_out.txt')

    print("Getting show_info data: %s" % url)
    urllib.request.urlretrieve(url, show_info_file)
    urllib.request.urlcleanup()

    with open(show_info_file, "r") as file_to_read:
        for line in file_to_read:
            if "spikes=" in line:
                spikes_path_lines= line.strip()
            elif "t_rec=" in line:
                t_rec_line= line.strip()


    print(spikes_path_lines)
    print(t_rec_line)
    # parsing sums path
    sums_path    = spikes_path_lines.replace('spikes=', '')
    t_rec_actual = t_rec_line.replace('t_rec=', '')

    # Build the url of the fits file using the SUMS path
    fits_url = JSOC + sums_path
    # Local file path
    filename = 'spikes_' + t_rec_actual.replace('-','_').replace(':','_')+ '_%s'%wavelength + '.fits'
    filepath = os.path.join(localpath, filename)

    return fits_url, filepath

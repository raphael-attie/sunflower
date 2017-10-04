import drms
import os
import urllib

# Local directory for download. E.g: ~/Data
download_dir = os.environ['HOME'] + '/Data'
# Instantiate a drms client
c = drms.Client(email='raphael.attie@nasa.gov', verbose=True)
# Request an export. Example with just 10 minutes of spikes for 171
r_171 = c.export('aia.lev1_euv_12s[2010.05.13_00:00/1h][171]{spikes}', method='url-tar', protocol='fits')
r_171.wait()

if r_171.status != 0:
    # If this is in a function, consider doing a sys.exit() to issue an error message and exit.
    print("Problem with the export request")
else:
    ##  Download using the drms module. See below for alternate download method
    # r_171.download(download_dir)

    ## If needed, get the url of the tar file (for download at a later time with other download libraries like "urllib")
    url_171, filename = r_171.urls['url'][0], r_171.urls['filename'][0]
    # Build a local file path using the exported tar filename
    local_filepath = os.path.join(download_dir, filename)
    # Download using urllib.
    urllib.request.urlretrieve(url_171, local_filepath)
    # Clear the cache that may have built up
    urllib.request.urlcleanup()


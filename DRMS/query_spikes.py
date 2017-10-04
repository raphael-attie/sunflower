import drms

# Local directory for download
download_dir = '~/Data'
# Instantiate a drms client
c = drms.Client(email='raphael.attie@nasa.gov', verbose=True)
# Request an export. Example with just 10 minutes of spikes for 171
r_171 = c.export('aia.lev1_euv_12s[2010.05.13_00:00/1h][171]{spikes}', method='url-tar', protocol='fits')
r_171.wait()

if r_171.status != 0:
    ## If needed, get the url of the tar file (for download at a later time with other download libraries like "urllib")
    # url_171 = r_171.urls['url'][0]

    ## Download using the drms module
    r_171.download(download_dir)

else:
    print("Problem with the export request")

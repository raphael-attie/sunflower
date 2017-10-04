import drms
import os
import urllib
import logging

def setup_logger(output_directory):
    # Setup logging
    logger = logging.getLogger('query_spikes')
    logger.setLevel(logging.INFO)
    # create file handler and set level to debug
    logfilepath = os.path.join(output_directory, 'spikes_download.log')
    fh = logging.FileHandler(filename=logfilepath, mode='w')
    fh.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to file handler
    fh.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(fh)
    return logger


# Local directory for download and log file. E.g: ~/Data
download_dir = os.path.join(os.environ['HOME'], 'Data')

# Instantiate the logger. See function def above
logger_spikes = setup_logger(download_dir)

# Instantiate a drms client
c = drms.Client(email='raphael.attie@nasa.gov', verbose=True)

# Request an export. Example with just 10 minutes of spikes for 171
req = c.export('aia.lev1_euv_12s[2010.05.16_00:00/1h][171]{spikes}', method='url-tar', protocol='fits')
req.wait()

if req.status != 0:
    # If this is in a function, consider doing a sys.exit() to issue an error message and exit.
    print("Problem with the export request")
else:
    ##  Download using the drms module. See below for alternate download method
    # req.download(download_dir)

    ## If needed, get the url of the tar file (for download at a later time with other download libraries like "urllib")
    url, filename = req.urls['url'][0], req.urls['filename'][0]
    # Build a local file path using the exported tar filename
    local_filepath = os.path.join(download_dir, filename)
    # Download using urllib.
    try:
        urllib.request.urlretrieve(url, local_filepath)
        logger_spikes.info('Downloaded: %s' % url)
    except Exception as err:
        logger_spikes.error('Download failed at: %s with error: %s'%(url, err))
        print('Download failed at: %s'%url)
    finally:
        print('Cleaning up...')
        # Clear the cache that may have built up
        urllib.request.urlcleanup()
        # Close the log file handler(s) and remove each one from the logger.
        _ = [handler.close() for handler in logger_spikes.handlers]
        _ = [logger_spikes.removeHandler(handler) for handler in logger_spikes.handlers]



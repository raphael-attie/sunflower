def get_harvey_lon(date, radians=False):
    """
    Need to update the rotation period to Carrington (~27-ish days) and the Carrington reference date, which is currently is the number:
    2415023.5 in julian day. The date needs to be of type  astropy.time.core.Time
    :param date:
    :param radians:
    :return:
    """
    # 2415023.5 JD = Jan 4, 1900 => 1st Harvey Rotation
    # 1 Harvey Rotation => 360 degrees in 33 days



    if not isinstance(date, astropy.time.core.Time):
        raise ValueError('Input needs to be an astropy time object.')

    if radians:
        return Longitude([math.radians(((360. / 33.) * (date.jd - 2415023.5)) - (np.floor(((360. / 33.) * (date.jd - 2415023.5)) / 360.) * 360.))] * u.rad)
    else:
        return Longitude([((360. / 33.) * (date.jd - 2415023.5)) - (np.floor(((360. / 33.) * (date.jd - 2415023.5)) / 360.) * 360.)] * u.deg)
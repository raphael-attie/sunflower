"""
Provides more advanced functions for queries with the DRMS and data displays
"""


from astropy import units as u

def hg_overlay(axes, color='white'):
    """
    Create a heliographic overlay using wcsaxes.
    Also draw a grid and label the top axes.
    Parameters
    ----------
    axes : `wcsaxes.WCSAxes` object.
        The `~wcsaxes.WCSAxes` object to create the HGS overlay on.
    color : color of the grid and ticks
    Returns
    -------
    overlay : wcsaxes overlay
        The overlay object.
    """
    #overlay = axes.get_coords_overlay('heliographic_stonyhurst')
    overlay = axes.get_coords_overlay('heliographic_carrington')

    lon = overlay[0]
    lat = overlay[1]

    lon.coord_wrap = 180
    lon.set_major_formatter('dd')

    lon.set_axislabel('Solar Longitude')
    lat.set_axislabel('Solar Latitude')

    lon.set_ticks(spacing=10. * u.deg, color=color)
    lat.set_ticks(spacing=10. * u.deg, color=color)

    overlay.grid(color=color, alpha=0.5)

    return overlay



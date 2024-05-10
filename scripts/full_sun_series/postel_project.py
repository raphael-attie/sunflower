import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map.header_helper import make_heliographic_header
import sunpy.map
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

hmif = sorted(Path('/Users/rattie/data/HMI/full_sun/2023_11_17').glob('*.fits'))
out_shape = [1024, 1024]

for i, f in enumerate(hmif[0:181]):
    hmi_map = sunpy.map.Map(f)

    origin_carr = SkyCoord(100.347954 * u.deg, 50 * u.deg, frame=frames.HeliographicCarrington,
                            obstime=hmi_map.date, observer='Earth')
    # origin_stonyhurst = origin_carr2.transform_to(frames.HeliographicStonyhurst)

    out_header = sunpy.map.make_fitswcs_header(
        out_shape,
        origin_carr,
        instrument='AIA',
        observatory='SDO',
        scale=[0.0301, 0.0301] * u.deg / u.pix,
        projection_code="ARC"
    )

    out_map = hmi_map.reproject_to(out_header)
    out_map.save(f'/Users/rattie/data/HMI/full_sun/2023_11_17/postel_x0_lat50/hmi_map_{i:04d}.fits', overwrite=True)

    hmi_rotated = hmi_map.rotate(order=3)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 2, 1, projection=hmi_rotated)
    hmi_rotated.plot(axes=ax)
    hmi_rotated.draw_grid(axes=ax, color='blue')
    hmi_rotated.draw_limb(axes=ax, color='blue')
    ax.plot_coord(origin_carr, 'o', color='red', fillstyle='none', markersize=20)

    ax = fig.add_subplot(1, 2, 2, projection=out_map)
    out_map.plot(axes=ax)
    out_map.draw_grid(axes=ax, color='blue')
    out_map.draw_limb(axes=ax, color='blue')
    ax.plot_coord(origin_carr, 'o', color='red', fillstyle='none', markersize=20)
    ax.set_title('Postel projection centered at ROI', y=-0.1)
    plt.savefig(f'/Users/rattie/data/HMI/full_sun/2023_11_17/postel_x0_lat50/figs/hmi_map_{i:04d}.jpg')
    plt.close()
#
# for i, f in enumerate(hmif[0:181]):
#     hmi_map = sunpy.map.Map(f)
#
#     origin_carr = SkyCoord(100.347954 * u.deg, -50 * u.deg, frame=frames.HeliographicCarrington,
#                             obstime=hmi_map.date, observer='Earth')
#     # origin_stonyhurst = origin_carr2.transform_to(frames.HeliographicStonyhurst)
#
#     out_header = sunpy.map.make_fitswcs_header(
#         out_shape,
#         origin_carr,
#         instrument='AIA',
#         observatory='SDO',
#         scale=[0.0301, 0.0301] * u.deg / u.pix,
#         projection_code="ARC"
#     )
#
#     out_map = hmi_map.reproject_to(out_header)
#     out_map.save(f'/Users/rattie/data/HMI/full_sun/2023_11_17/postel_x0_latm50/hmi_map_{i:04d}.fits', overwrite=True)

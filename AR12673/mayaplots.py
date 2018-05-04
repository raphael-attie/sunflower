import numpy as np
from mayavi import mlab
from matplotlib.colors import Normalize

px_meter = int(0.03 * 3.1415/180 * 6.957e8)
ms_unit = int(px_meter / 45)

def plot_im(im, vmin=None, vmax=None):

    if vmin is None:
        vmin = im.min()
    if vmax is None:
        vmax = im.max()

    im_t = im.copy().T
    dims = im_t.shape[0], im_t.shape[1]
    s1 = mlab.imshow(im_t, vmin=vmin, vmax=vmax, colormap='Greys', extent=[0, dims[1] - 1, 0, dims[0] - 1, 0, 0])
    s1.module_manager.scalar_lut_manager.reverse_lut = True
    s1.update_pipeline()
    return s1


def plot_lanes(lanes):

    lanes_t = lanes.copy().T
    lanes_norm = Normalize(0, 0.7 * lanes_t.max(), clip=True)(lanes_t)

    dims = lanes_norm.shape[0], lanes_norm.shape[1]
    s2 = mlab.imshow(lanes_norm, colormap='Blues', extent=[0, dims[1] - 1, 0, dims[0] - 1, 0, 0])
    cmap = s2.module_manager.scalar_lut_manager.lut.table.to_array()
    cmap[:, -1] = np.linspace(0, 255, 256)
    s2.module_manager.scalar_lut_manager.lut.table = cmap
    mlab.view(0, 0)
    s2.update_pipeline()
    return s2


def plot_flow_vectors(vx, vy, fig, offset=0, reverse = False):

    vx_t = vx.copy().T
    vy_t = vy.copy().T
    vz_t = np.zeros(vx_t.shape)

    u = np.zeros(vx_t.shape + (2,))
    v = np.zeros(vx_t.shape + (2,))
    w = np.zeros(vx_t.shape + (2,))
    u[..., 0] = vx_t *ms_unit #/ norm.max()
    v[..., 0] = vy_t *ms_unit #/ norm.max()
    w[..., 0] = vz_t


    x, y, z = np.mgrid[0:vx_t.shape[0], 0:vy_t.shape[1], 0:2]

    # Possible bug here if using a recent version of numpy > 1.11
    # See https://github.com/enthought/mayavi/issues/499
    src = mlab.pipeline.vector_field(x+offset, y+offset, z, u, v, w, figure=fig)
    #src = mlab.pipeline.vector_field(u, v, w, figure=fig)
    magnitude = mlab.pipeline.extract_vector_norm(src)
    vec = mlab.pipeline.vectors(magnitude, mask_points=20, scale_factor=8., line_width=4, colormap='Oranges')
    vec.module_manager.vector_lut_manager.reverse_lut = reverse
    vec.glyph.glyph_source.glyph_position = 'center'

    # TODO: Change the range of the colorbar and the number of labels to have round numbers

    vec.update_pipeline()
    mlab.view(0, 0)
    #mlab.move(-100, 0, 0)

    return magnitude, vec


def plot_streamlines(magnitude, vx, vy, reverse = False):

    vx_t = vx.copy().T
    vy_t = vy.copy().T

    flow = mlab.pipeline.streamline(magnitude, seedtype='plane', seed_visible=False, seed_scale=0.5, seed_resolution=12,
                                    linetype='line', line_width=2, colormap='Oranges')
    flow.seed.widget.normal_to_z_axis_ = 1
    flow.seed.widget.normal_to_z_axis = 1
    flow.seed.widget.center = np.array([(vx_t.shape[0])/2+1, (vy_t.shape[1])/2+1, 0])
    flow.seed.widget.origin = np.array([0, 0, 0])
    flow.seed.widget.point1 = np.array([vx_t.shape[0]-1, 0, 0])
    flow.seed.widget.point2 = np.array([0, vy_t.shape[1]-1, 0])
    flow.seed.widget.enabled = 1
    #flow.seed.widget.resolution = 12
    flow.stream_tracer.maximum_propagation = 30
    flow.stream_tracer.integration_direction = 'both'
    flow.module_manager.scalar_lut_manager.reverse_lut = reverse
    flow.module_manager.scalar_lut_manager.use_default_range=False


    return flow


def add_scalar_colorbar(src, data_min, data_max, nlabels):

    #mlab.move(-380, 0, 0)


    src.module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([0.87, 0.14])
    src.module_manager.scalar_lut_manager.scalar_bar_representation.position2 = np.array([0.1, 0.725])
    src.module_manager.scalar_lut_manager.show_scalar_bar = True
    src.module_manager.scalar_lut_manager.show_legend = True

    src.module_manager.scalar_lut_manager.data_range = np.array([data_min, data_max])
    src.module_manager.scalar_lut_manager.number_of_labels = nlabels
    src.module_manager.scalar_lut_manager.data_name = 'v [m/s]'
    #module_manager2.scalar_lut_manager.title_text_property.shadow_offset = array([1, -1])
    src.module_manager.scalar_lut_manager.title_text_property.font_size = 10

    src.seed.update_pipeline()
    src.update_pipeline()
    src.seed.widget.enabled = 0
    src.seed.update_pipeline()


def add_vector_colorbar(src):

    mlab.move(-380, 0, 0)

    src.module_manager.vector_lut_manager.scalar_bar_representation.position = np.array([0.87, 0.14])
    src.module_manager.vector_lut_manager.scalar_bar_representation.position2 = np.array([0.1, 0.725])
    src.module_manager.vector_lut_manager.show_scalar_bar = True

    #src.seed.update_pipeline()
    src.update_pipeline()
    src.module_manager.update()
    #src.seed.widget.enabled = 0
    #src.seed.update_pipeline()


def add_axes_labels(fig, shape, ranges = None):


    ax = mlab.axes(figure = fig, xlabel='x', ylabel='y', extent=[0, shape[1]-1, 0, shape[0]-1, 0, 1], ranges= ranges, nb_labels=11,
                   z_axis_visibility=False, color=(0, 0, 0))
    ax._label_text_property.color = (0, 0, 0)
    ax._title_text_property.color = (0, 0, 0)
    ax.axes.label_format = '%.0f'
    ax.axes.font_factor = 1.0
    return ax

from bokeh.plotting import figure, curdoc
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, BoxEditTool, Div
from bokeh.plotting import show
import pickle

source0 = ColumnDataSource(data=dict(
    x0=[1, 2, 3, 4, 5],
    y0=[2, 5, 8, 2, 7],
))

tools = ['hover', 'reset']

plot = figure(x_axis_label='x',
              y_axis_label='y)',
              title="Test BoxEditTool",
              tools=tools)

source = ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})

r1 = plot.rect('x', 'y', 'width', 'height', source=source)
tool = BoxEditTool(renderers=[r1], num_objects=1)
plot.add_tools(tool)

div = Div(text='Box Info:')


def getBoxDims(attrname, old, new):
    xl = source.data['x'][0] - source.data['width'][0] / 2
    xr = source.data['x'][0] + source.data['width'][0] / 2
    yl = source.data['y'][0] - source.data['height'][0] / 2
    yu = source.data['y'][0] + source.data['height'][0] / 2
    div.text = 'Box Info: Xdims: ' + str(xl) + ',' + str(xr) + ', Ydims: ' + str(yl) + ',' + str(yu)
    rect_dict = {'x': source.data['x'][0], 'y': source.data['y'][0],
                 'width': source.data['width'][0], 'height': source.data['height'][0]}
    with open('source_data.pkl', 'wb') as f:
        pickle.dump(rect_dict, f)


source.on_change('data', getBoxDims)

# Add the data renderer (here a scatter plot)
scatter = plot.scatter(x='x0', y='y0', source=source0)

curdoc().add_root(row([plot, div]))
from matplotlib import pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy import asarray
from numpy.typing import NDArray


def rasterize_as_array(fig: pyplot.Figure) -> NDArray:
    """
    Rasterize a figure as a numpy array, which can be displayed on another axis using
    pyplot.imshow().
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    arr = asarray(canvas.buffer_rgba())
    return arr

from io import BytesIO

from matplotlib import pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy import asarray, array
from numpy.typing import NDArray
from PIL.Image import open as open_image


def rasterize_fig_as_array(fig: pyplot.Figure) -> NDArray:
    """
    Rasterize a figure as a numpy array, which can be displayed on another axis using
    pyplot.imshow().
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    arr = asarray(canvas.buffer_rgba())
    return arr


def rasterize_axes_as_array(ax: pyplot.Axes) -> NDArray:
    """
    Rasterize an axes as a numpy array, which can be displayed on another axis using
    pyplot.imshow().
    """
    fig = ax.figure
    # Ensure the renderer is updated
    fig.canvas.draw()

    # Convert figure to array
    with BytesIO() as buffer:
        fig.savefig(buffer, format='png', dpi=fig.dpi, transparent=True)
        buffer.seek(0)
        img_arr = array(open_image(buffer))

    bbox = ax.get_window_extent()
    x0, y0, x1, y1 = map(int, [bbox.x0, bbox.y0, bbox.x1, bbox.y1])

    # The y-axis in images is top-down but bbox y is bottom-up, so we need to flip
    height = img_arr.shape[0]
    cropped_img_arr = img_arr[height - y1: height - y0, x0:x1]

    return cropped_img_arr

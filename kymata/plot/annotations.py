from numpy import absolute, argmin


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    Thanks https://stackoverflow.com/a/34018322

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
     color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
     position = xdata.mean()
    # find the closest index
    start_ind = argmin(absolute(xdata - position))
    if direction == 'right':
     end_ind = start_ind + 1
    else:
     end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size,
                       )

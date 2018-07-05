import matplotlib.pyplot as plt
import numpy as np

def line_polar(center, theta, rmin, rmax):
    """ Calculate the coordinates of start and end point (x1,y1), (x2, y2) of a line in polar coordinates,
    given the center of the polar coordinate system, an angle (theta) and a length (radius).

    Example:

    data = np.zeros([512, 512])
    data[252:260, 252:260] = 1
    xp, yp = line_polar((256, 256), 30, 200)

    plt.figure(0)
    plt.imshow(data, origin='lower', cmap='gray')
    l2 = mlines.Line2D(xp, yp, color='yellow')
    plt.gca().add_line(l2)

    :param center: coordinate of the center of the polar reference frame
    :param theta: angle in degrees
    :param radius: length of the line from the center
    :return: cartesian coordinates of the 2 points necessary to draw the line.

    """

    xc, yc = center
    x1 = xc + rmin * np.cos(theta * np.pi/180)
    y1 = yc + rmin * np.sin(theta * np.pi/180)

    x2 = xc + rmax * np.cos(theta * np.pi / 180)
    y2 = yc + rmax * np.sin(theta * np.pi / 180)

    return (x1,x2), (y1,y2)



"""Cornerpoint grids."""


class CornerpointGrid (dict):  #pylint: disable=too-few-public-methods
    """Structure that describes a corner-point grid."""

    # object can be accessed either through attributes, or as dict
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__ (self, dimens, coord, zcorn, actnum):
        """Initialize a cornerpoint grid from parts.

        :param dimens: Dimensions of the grid; (ni, nj, nk)
        :type dimens:  :class:`numpy.ndarray`,
                       dtype = numpy.int32,
                       shape = (3,)
        :param coord:  Pillar coordinates
        :type coord:   :class:`numpy.ndarray`,
                       dtype = numpy.float32,
                       shape = (nj + 1, ni + 1, 2, 3)
        :param zcorn:  Corner depths
        :type zcorn:   :class:`numpy.ndarray`
                       dtype = numpy.float32,
                       shape = (nk, 2, nj, 2, ni, 2)
        :param actnum: Active cell flags
        :type actnum:  :class:`numpy.ndarray`
                       dtype = numpy.bool,
                       shape = (nk, nj, ni)

        .. note::
        Dimensions are specified in 'Fortran' order, which is reverse
        of what the shape of the array is (by default) in NumPy.
        """
        super (CornerpointGrid, self).__init__ ()

        # store these properties in the dictionary
        self.dimens = dimens
        self.coord  = coord
        self.zcorn  = zcorn
        self.actnum = actnum

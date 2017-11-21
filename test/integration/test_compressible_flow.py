import unittest
import numpy as np

from porepy.numerics.compressible import problems
from porepy.grids import structured
from porepy.fracs import meshing
from porepy.params import tensor


class TestCompressibleFlow(unittest.TestCase):

    def test_3d_Fracture_problem(self):
        problem = UnitSquareInjectionMultiDim()
        solution = problem.solve()
        p = solution['flow'][-1]
        problem.time_disc().split(problem.grid(), 'pressure', p)
        for g, d in problem.grid():
            if g.dim == 3:
                pT = d['pressure']
        p_refT = _reference_solution_multi_grid()

        assert np.allclose(pT, p_refT)


###############################################################################


class MatrixDomain(problems.SlightlyCompressibleData):
    def __init__(self, g, d):
        problems.SlightlyCompressibleData.__init__(self, g, d)


class FractureDomain(problems.SlightlyCompressibleData):
    def __init__(self, g, d):
        problems.SlightlyCompressibleData.__init__(self, g, d)
        aperture = np.power(0.001, 3 - g.dim)
        self.data()['param'].set_aperture(aperture)

    def permeability(self):
        kxx = 1000 * np.ones(self.grid().num_cells)
        return tensor.SecondOrder(3, kxx)


class IntersectionDomain(FractureDomain):
    def __init__(self, g, d):
        FractureDomain.__init__(self, g, d)

    def source(self, t):
        assert self.grid().num_cells == 1, '0D grid should only have 1 cell'
        f = .4 * self.grid().cell_volumes  # m**3/s
        return f * (t < .05)


def set_sub_problems(gb):
    gb.add_node_props(['problem'])
    for g, d in gb:
        if g.dim == 3:
            d['problem'] = MatrixDomain(g, d)
        elif g.dim == 2 or g.dim == 1:
            d['problem'] = FractureDomain(g, d)
        elif g.dim == 0:
            d['problem'] = IntersectionDomain(g, d)
        else:
            raise ValueError('Unkown grid-dimension %d' % g.dim)


class UnitSquareInjectionMultiDim(problems.SlightlyCompressible):

    def __init__(self):

        f_1 = np.array([[0, 1, 1, 0], [.5, .5, .5, .5], [0, 0, 1, 1]])
        f_2 = np.array([[.5, .5, .5, .5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f_3 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [.5, .5, .5, .5]])
        f_set = [f_1, f_2, f_3]

        g = meshing.cart_grid(f_set, [4, 4, 4], physdims=[1, 1, 1])
        g.compute_geometry()
        g.assign_node_ordering()
        set_sub_problems(g)
        self.g = g
        # Initialize base class
        problems.SlightlyCompressible.__init__(self, self.g, 'flow')

    #--------grid function--------

    def grid(self):
        return self.g

    #--------Time stepping------------

    def time_step(self):
        return .005

    def end_time(self):
        return 0.05

###############################################################################


def _reference_solution_multi_grid():
    pT = np.array([0.00983112,  0.01637886,  0.01637886,  0.00983112,  0.01637886,
                   0.02264049,  0.02264049,  0.01637886,  0.01637886,  0.02264049,
                   0.02264049,  0.01637886,  0.00983112,  0.01637886,  0.01637886,
                   0.00983112,  0.01637886,  0.02264049,  0.02264049,  0.01637886,
                   0.02264049,  0.03234641,  0.03234641,  0.02264049,  0.02264049,
                   0.03234641,  0.03234641,  0.02264049,  0.01637886,  0.02264049,
                   0.02264049,  0.01637886,  0.01637886,  0.02264049,  0.02264049,
                   0.01637886,  0.02264049,  0.03234641,  0.03234641,  0.02264049,
                   0.02264049,  0.03234641,  0.03234641,  0.02264049,  0.01637886,
                   0.02264049,  0.02264049,  0.01637886,  0.00983112,  0.01637886,
                   0.01637886,  0.00983112,  0.01637886,  0.02264049,  0.02264049,
                   0.01637886,  0.01637886,  0.02264049,  0.02264049,  0.01637886,
                   0.00983112,  0.01637886,  0.01637886,  0.00983112])
    return pT

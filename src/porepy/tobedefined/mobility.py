import porepy as pp
from typing import Callable

import pdb

"""
"""


def mobility(relative_permeability: Callable) -> pp.ad.AdArray:
    """ """

    def mob(saturation, dynamic_viscosity, relative_permeability=relative_permeability):
        return relative_permeability(saturation) / dynamic_viscosity

    return mob


def mobility_operator(mobility: Callable) -> pp.ad.Operator:
    """ """

    def mob_op(subdomains, saturation, dynamic_viscosity, mobility=mobility):
        s = saturation(subdomains)
        mob_operator = pp.ad.Function(mobility, "mobility_operator")
        mob = mob_operator(s, pp.ad.Scalar(dynamic_viscosity))
        return mob

    return mob_op

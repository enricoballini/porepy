import porepy as pp


def mobility(saturation, dynamic_viscosity):
    """ """
    mob = pp.rel_perm_brooks_corey(saturation) / dynamic_viscosity
    return mob


def mobility_operator(subdomains, saturation, dynamic_viscosity):
    """ """
    s = saturation(subdomains)
    mobility_operator = pp.ad.Function(mobility, "mobility_operator")
    mob = mobility_operator(s, pp.ad.Scalar(dynamic_viscosity))
    return mob

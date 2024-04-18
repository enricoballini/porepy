import numpy as np

"""
- pay attention, here we compute the number EA that appears in the equation taking values from code, 
 so d/dt + div( ... + EA ... ) = 0
 - in the code I always solve eqtuaion with EA = 1. The scaling values you find there are the so-called (at leat by SU2 group) "scaling values" and the values you find here are the "reference values"
 - nobody has to know about scaling values, they are just numerical values to make the simulation work. Reference values are necessary to compare different simulations. They appear in the paper.


"""


def eb_ar_number(dynamic_viscosity_0, Ka_0, rho_0, L_0, phi_0, gravity_0):
    """ """
    eb_number = rho_0**2 * gravity_0 * L_0 * Ka_0 * phi_0 / dynamic_viscosity_0**2
    return eb_number


# case 1 horizontal matrix
phi_0 = 0.25
L_0 = 1
gravity_0 = 1
dynamic_viscosity_0 = 1
rho_0 = 0.5  # |rho_phase_0-rho_phase_1|
Ka_0 = 1
ebar = eb_ar_number(dynamic_viscosity_0, Ka_0, rho_0, L_0, phi_0, gravity_0)
print("case 1 horizontal EA = ", ebar)

# case 1 vertical matrix
phi_0 = 0.25
L_0 = 1
gravity_0 = 1
dynamic_viscosity_0 = 1
rho_0 = 0.5
Ka_0 = 1
ebar = eb_ar_number(dynamic_viscosity_0, Ka_0, rho_0, L_0, phi_0, gravity_0)
print("case 1 vertical EA = ", ebar)

# case 1 slanted matrix
phi_0 = 0.25
L_0 = 1
gravity_0 = 1
dynamic_viscosity_0 = 1
rho_0 = 0.5
Ka_0 = 1
ebar = eb_ar_number(dynamic_viscosity_0, Ka_0, rho_0, L_0, phi_0, gravity_0)
print("case 1 slanted EA = ", ebar)

# case 2
phi_0 = 0.25
L_0 = 1
gravity_0 = 1
dynamic_viscosity_0 = 1
rho_0 = 0.5
Ka_0 = 1e2
ebar = eb_ar_number(dynamic_viscosity_0, Ka_0, rho_0, L_0, phi_0, gravity_0)
print("case 2 EA = ", ebar)

# case 3 matrix
phi_0 = 0.2
L_0 = 2.25
gravity_0 = 1
dynamic_viscosity_0 = 1
rho_0 = 0.5
Ka_0 = 1e2
ebar = eb_ar_number(dynamic_viscosity_0, Ka_0, rho_0, L_0, phi_0, gravity_0)
print("case 3 matrix EA = ", ebar)

# case 3 fracture
phi_0 = 0.2
L_0 = 1.75
gravity_0 = 1
dynamic_viscosity_0 = 1
rho_0 = 0.5
Ka_0 = 1e4
ebar = eb_ar_number(dynamic_viscosity_0, Ka_0, rho_0, L_0, phi_0, gravity_0)
print("case 3 fracture EA = ", ebar)

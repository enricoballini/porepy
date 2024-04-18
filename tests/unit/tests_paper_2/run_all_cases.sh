#!/bin/sh

# NOTE: launch this code in the folder where it is located (there are some imports hardocded in the cases)

export OMP_NUM_THREADS=2

# Case 1 horizontal 
python3 case_1_horizontal_hu.py
python3 case_1_horizontal_ppu.py
python3 plot_flip_flop.py 1 horizontal
python3 plot_chops.py 1 horizontal
python3 plot_newton.py 1 horizontal
python3 plot_mass.py 1 horizontal
# python3 plot_beta.py 1 horizontal

# Case 1 vertical
python3 case_1_vertical_hu.py
python3 case_1_vertical_ppu.py
python3 plot_flip_flop.py 1 vertical
python3 plot_chops.py 1 vertical
python3 plot_newton.py 1 vertical
# python3 plot_beta.py 1 vertical

# Case 1 slanted non conforming
python3 case_1_slanted_hu.py
python3 case_1_slanted_ppu.py
python3 plot_flip_flop.py 1 slanted_non_conforming
python3 plot_chops.py 1 slanted_non_conforming
python3 plot_newton.py 1 slanted_non_conforming
# python3 plot_beta.py 1 slanted_non_conforming

# # Case 1 slanted convergence
# python3 case_1_slanted_hu_convergence.py
# # python3 case_1_slanted_ppu_convergence.py # don't run the grid convergence of ppu
# python3 case_1_slanted_hu_convergence_non_conf.py
# python3 plot_convergence.py conforming
# python3 plot_convergence.py non_conforming

# # Case 2
# python3 case_2_hu.py
# python3 case_2_ppu.py
# python3 plot_flip_flop.py 2 
# python3 plot_chops.py 2
# python3 plot_newton.py 2
# python3 plot_beta.py 2

# # Case 3
# python3 case_3_hu.py
# python3 case_3_ppu.py
# python3 plot_flip_flop.py 3
# python3 plot_chops.py 3
# python3 plot_newton.py 3








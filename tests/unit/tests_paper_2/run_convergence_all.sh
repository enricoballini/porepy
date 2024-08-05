#!/bin/sh

python3 case_1_slanted_hu_convergence.py
# python3 case_1_slanted_ppu_convergence.py # don't run the grid convergence of ppu
python3 case_1_slanted_hu_convergence_non_conf.py
python3 plot_convergence_alessio.py

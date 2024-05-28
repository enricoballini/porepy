#!/bin/bash

for ((i=1;i<=3;i++));
do
    cd RUN_$i
        eclrun --queuesystem=PBSPRO -q eclipse eclipse GUE2022_UPP2_TESI.DATA
    cd ..
done


# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:01:48 2021

@author: Zhang donghao
"""

import random
import math
n=7001
while(n<=7001):
    with open("test%d.in"%n,"w") as f:
        f.write(
                '#material: 9 0.001 1 0 concrete\n'
                '#material: 20 0.01 1 0 soil\n'
                '#domain: 2.0 4.0 0.4\n'
                '#dx_dy_dz: 0.005 0.005 0.005\n'
                '#time_window: 50e-9\n'
                '#box: 0.000 0.000 0.000 2.000 4.000 0.4 concrete\n'
                '#waveform: ricker 1 5.0e8 my_ricker\n'
                '#hertzian_dipole: z 0.10 0.2 0.2 my_ricker\n'
                '#rx_array: 1.9 0.2 0.2 1.9 0.2 0.2 0 0 0\n'
				'#rx_steps: 0 0.1 0\n'
                '#src_steps: 0 0.1 0\n'
                )
    void_x=random.uniform(0.2,1.8)
    void_y=random.uniform(0.5,3.5)
    void_r1=random.uniform(0.07,0.2)
    with open("test%d.in"%n,"a+") as f:
        f.write('#cylinder: %.2f %.2f 0.05 %.2f %.2f 0.35 %.2f soil\n'%(void_x,void_y,void_x,void_y,void_r1)
               )
    with open("test%d.in"%n,"a+") as f:
        f.write(
                '#pml_cells: 10 10 10 10 10 10\n')
    n=n+1


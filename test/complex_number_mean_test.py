"""Test of complex number transformation 
to then sum and calculate the mean
"""

#%% 
import pandas as pd
import numpy as np
from IPython.display import display



angles=[-135,45]

# prueba=behaviour_filtered_df.loc[0:1,['deg_head_cricket']]
prueba=pd.DataFrame({'angle_head_cricket': np.deg2rad(np.array(angles))}) ## important work with angles in radians
prueba['complex']=np.cos(prueba['angle_head_cricket'])+  1j *np.sin(prueba['angle_head_cricket'])
prueba['angle_from_complex']=np.angle(prueba['complex'],deg=True)
display(prueba)
def circular_mean_angle(angles:pd.Series):
    """Function to calculate circular mean from complex numbers

    Args:
        angles (pd.Series): angels in radians, usually betwwen (pi,-pi) for more coherent stuff. I want 
        oposite angles to give an angle of 0. However absolute avlues of -45 or -135 are really strange.

    Returns:
        _type_: _description_
    """
    suma_com=np.sum(angles)
    final_angle=np.angle(suma_com,deg=True)
    final_magnitude=np.abs(suma_com)

    print(f'sum complex numbers: {suma_com}')
    print(f'final angle: {final_angle}')
    print(f'final magnitude: {final_magnitude}')

    return final_angle,final_magnitude

circular_mean_angle(prueba['complex'])

# %%



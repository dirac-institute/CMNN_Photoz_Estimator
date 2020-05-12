import os
import numpy as np

def convert_visits_to_depths( visits, m5_single = [23.9, 25.0, 24.7, 24.0, 23.3, 22.1] ):
    ### Convert from number of visits per filter to the 5-sigma limiting magnitude depths

    ### Inputs
    ###   visits    : number of visits in ugrizy
    ###   m5_single : single standard (30 sec) visit depth in ugrizy

    ### Outputs
    ###   depths : 5-sigma limiting magnitudes in ugrizy

    if len(visits) != 6:
        print('Error. Input visits must have six elements.')
        print('  len(visits) : %i ' % len(visits))
        print('Exit (bad inputs).')
        exit()

    m5s = np.asarray( m5_single, dtype='float' )
    Nvis = np.asarray( visits, dtype='float' )

    depths = m5s + ( 1.25 * np.log10( Nvis ) )

    return depths


def convert_year_to_depths_baseline( year, BL_visits_10yr = [56, 80, 184, 184, 160, 160], \
    m5_single = [23.9, 25.0, 24.7, 24.0, 23.3, 22.1] ):
    ### Convert from the year of the survey to the 5-sigma limiting magnitude depths assuming a baseline strategy

    ### Input
    ###   year           : year of survey
    ###   BL_visits_10yr : visits at 10 years for a baseline observing strategy in ugrizy
    ###   m5_single      : single standard (30 sec) visit depth in ugrizy

    ### Outputs
    ###   depths : 5-sigma limiting magnitudes in ugrizy

    m5s = np.asarray( m5_single, dtype='float' )
    BLv = np.asarray( BL_visits_10yr, dtype='float' )
    yr  = float(year)

    depths = m5s + ( 1.25 * np.log10( BLv * yr / 10.0 ) )
    
    return depths

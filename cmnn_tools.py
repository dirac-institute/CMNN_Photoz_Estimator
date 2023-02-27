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


def calculate_magnitude_error( filt='i', truemag=25.0, m5=25.58, gammas=[0.037,0.038,0.039,0.039,0.04,0.04] ):
    ### Calculate the error in observed apparent magnitude.

    ### Input
    ###   filter  : one of 'u', 'g', 'r', 'i', 'z', or 'y' (default 'i')
    ###   truemag : galaxy true catalog magnitude in filter (default 25, typical i-band cut)
    ###   m5      : LSST 5-sigma limiting magnitude depth in filter (default 25.58, year 1 baseline depth in i)
    ###   gammas  : gamma values to use for LSST filters ugrizy (default values from Ivezic+2019)
    print( 'Inputs: ', filt, truemag, m5, gammas)

    ### Output
    ###   error : the error in observed apparent magnitude

    all_filts = np.asarray( ['u','g','r','i','z','y'], dtype='str' )
    fx = np.where( all_filts == filt )[0]
    if len(fx) != 1:
        print( 'Passed argument for filt must be one of: u, g, r, i, z, or y.' )
        print( 'Not recognized: filt = ', filt )
        print( 'Exit (bad value for filt).' )

    gamma = gammas[fx[0]]

    error = np.sqrt( ( 0.04 - gamma ) * ( np.power( 10.0, 0.4*( truemag - m5 ) ) ) + \
        gamma * ( np.power( 10.0, 0.4*( truemag - m5 ) )**2 ) )

    return error



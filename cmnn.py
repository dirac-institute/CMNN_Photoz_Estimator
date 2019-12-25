import os
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_photoz( test_c, test_ce, test_id, train_c, train_z, train_id, thresh_table, \
    ppf_value=0.68, selection_mode=2, minimum_Ncolors=3, minimum_Nneighbors=10 ):

    # This code will identify the color-matched nearest-neighbors (CMNN) subset of training galaxies
    # for a given test galaxy and return a photometric redshift and uncertainty.

    # Input definitions:
    #  test_c   = array of colors for test galaxy
    #  test_ce  = array of color errors for test galaxy
    #  test_id  = unique integer identifier for test galaxy
    #  train_c  = array of colors for all training-set galaxies
    #  train_z  = array of color errors for all training-set galaxies
    #  train_id = array of unique integer identifiers for all training-set galaxies
    #
    #  ppf_value          = percent point function value (typically 0.68 or 0.95)
    #  selection_mode     = how the photo-z will be selected from the CMNN subset of training galaxies
    #                       0 : random, 1 : nearest neighbor, 2 : weighted random
    #  minimum_Ncolors    = minimum number of colors used to identify the CMNN subset of training galaxies
    #  minimum_Nneighbors = the minimum size of the CMNN subset of training galaxies
    #  verbose            = write out progress statements (True or False)

    # Calculate the Mahalanobis Distance and the number of degrees of freedom for each training set galaxy
    MahalanobisDistance = np.nansum( ( test_c - train_c )**2 / test_ce**2, axis=1 )
    DegreesOfFreedom    = np.nansum( ( test_c - train_c ) / ( test_c - train_c ), axis=1 )

    # Determine the appropriate threshold that should apply to each training set galaxy
    #   Could do the slow way:  thresholds = chi2.ppf( ppf_value, DegreesOfFreedom )
    thresholds = np.zeros( len(train_c), dtype='float' )
    # Only calculate the thresholds for training galaxies that could conceivably be a CMNN
    tx = np.where( (DegreesOfFreedom >= minimum_Ncolors) & (MahalanobisDistance <= thresh_table[-1]) )[0]
    for i in tx:
        thresholds[i] = thresh_table[ DegreesOfFreedom[i] ]
    del tx

    # Identify the indicies of the CMNN subset of training-set galaxies
    index = np.where( \
        ( test_id != train_id ) & \
        ( DegreesOfFreedom >= minimum_Ncolors ) & \
        ( MahalanobisDistance > 0.0 ) & \
        ( MahalanobisDistance < thresholds ) )[0]

    # Determine the photometric redshift for this test galaxy
    # if there are a sufficient number of training-set galaxies in the CMNN subset
    if len(index) >= minimum_Nneighbors:
        # choose randomly from the color matched sample
        if selection_mode == 0:
            rval  = int(np.random.uniform(low=0, high=len(index)-1))
            rival = index[rval]
            Photoz      = train_z[rival]
            PhotozError = np.std( train_z[index] )
            del rval,rival
        # choose the nearest neighbor, the best color match
        if selection_mode == 1:
            tx = np.where( MahalanobisDistance[index] == np.nanmin(MahalanobisDistance[index]) )[0]
            if len(tx) == 1:
                rval = int(tx)
            if len(tx) > 1:
                # if there's more than one best match, choose randomly
                tval = np.random.uniform(low=0,high=len(tx)-1)
                rval = int(tx[tval])
                del tval
            rival = index[rval]
            Photoz      = train_z[rival]
            PhotozError = np.std( train_z[index] )
            del tx,rval,rival
        # weight by how good the color match is and then choose randomly
        if selection_mode == 2:
            tweight = float(1.00) / MahalanobisDistance[index]
            weight  = tweight / np.sum(tweight)
            rval    = np.random.choice( range(len(index)), size=1, replace=False, p=weight )
            rival   = index[rval]
            Photoz      = train_z[rival]
            PhotozError = np.std( train_z[index] )
            del tweight,weight,rval,rival

    # if there are too few training-set galaxies in the CMNN subset
    if len(index) < minimum_Nneighbors:
        index2 = np.where( (MahalanobisDistance > 0.0) & (DegreesOfFreedom >= minimum_Ncolors) )[0]
        tempMD = MahalanobisDistance[index2]
        tempTZ = train_z[index2]
        tempDF = DegreesOfFreedom[index2]
        # identify the nearest neighbors and use them as the CMNN subset
        sx = np.argsort( tempMD )
        new_MD = tempMD[sx[0:minimum_Nneighbors]]
        new_TZ = tempTZ[sx[0:minimum_Nneighbors]]
        new_DF = tempDF[sx[0:minimum_Nneighbors]]
        del index2,tempMD,tempTZ,tempDF,sx
        # calculate the new 'effective PPF' based on the most distant nearest neighbor
        new_ppf_value = chi2.cdf( new_MD[minimum_Nneighbors-1], new_DF[minimum_Nneighbors-1] )
        # inflate the photo-z error appropriately
        temp   = np.std( new_TZ )
        PhotozError = temp * (new_ppf_value / ppf_value)
        del temp,new_ppf_value
        # choose randomly from nearest dselect galaxies
        if selection_mode == 0:
            rval   = int( np.floor( np.random.uniform(low=0, high=len(new_TZ)-1) ) )
            Photoz = new_TZ[rval]
            del rval
        # choose nearest neighbour, use nearest dselect for error
        if selection_mode == 1:
            Photoz = new_TZ[0]
        # weight by how good the color match is and then select
        if selection_mode == 2:
            tweight = float(1.00) / new_MD
            weight = tweight / np.sum(tweight)
            cx     = np.random.choice( range(len(new_TZ)), size=1, replace=False, p=weight )
            Photoz = new_TZ[cx]
            del tweight,weight,cx
        del new_MD,new_TZ,new_DF

    return [Photoz, PhotozError, len(index)]


def run( input_catalog_filename, output_catalog_filename, number_of_filters, \
    number_of_test_galaxies=None, minimum_number_of_colors=3, use_ppf_value=0.68):

    # Description of inputs
    # input_catalog_filename   = a string naming the input catalog file
    # output_catalog_filename  = a string naming the output catalog file (will be overwritten)
    # number_of_test_galaxies  = number of test galaxies to calculate zphot for (None = all)
    # number_of_filters        = number of filters of photometry in the input catalog
    # minimum_number_of_colors = minimum number of colors a galaxy must have to get a photoz estimate

    Nf = number_of_filters
    # Input catalog must contain these columns:
    #  0               = unique integer identifier
    #  1               = true redshift
    #  2 : Nf+1        = observed apparent magnitudes
    #  Nf+2 : (2*Nf)+1 = observed apparent magnitude errors

    # Output catalog will contain these columns only for galaxies with sufficient number of colors:
    #  0 = unique integer identifier
    #  1 = true redshift
    #  2 = photometric redshift
    #  3 = photometric redshift error
    #  4 = number of CMNN training-set galaxies

    # Create threshold lookup table because chi2.ppf too slow
    # Use number_of_filters as the maximum number of degrees of freedom (b/c it will always be less than that)
    use_thresh_table = np.zeros( number_of_filters, dtype='float' )
    for i in range(number_of_filters):
        use_thresh_table[i] = chi2.ppf(use_ppf_value,i)

    # Read in the input catalog
    mag_cols  = np.arange(Nf)+2
    mage_cols = np.arange(Nf)+Nf+2
    all_id   = np.loadtxt( input_catalog_filename, dtype='float', usecols={0})
    all_z    = np.loadtxt( input_catalog_filename, dtype='float', usecols={1})
    all_mag  = np.loadtxt( input_catalog_filename, dtype='float', usecols=mag_cols)
    all_mage = np.loadtxt( input_catalog_filename, dtype='float', usecols=mage_cols)

    # Calculate colors for all galaxies
    all_col  = np.zeros( (len(all_id),Nf-1), dtype='float' )
    all_cole = np.zeros( (len(all_id),Nf-1), dtype='float' )
    for c in range(Nf-1):
        all_col[:,c]  = all_mag[:,c] - all_mag[:,c+1]
        all_cole[:,c] = np.sqrt( all_mage[:,c]**2 + all_mage[:,c+1]**2 )

    # Calculate the number of colors
    all_Nc = np.nansum( all_col/all_col, axis=1 )

    # Delete galaxies that do not have the minimum number of colors
    tx = np.where( all_Nc < minimum_number_of_colors )[0]
    np.delete( all_id, tx, axis=0)
    np.delete( all_z, tx, axis=0)
    np.delete( all_mag, tx, axis=0)
    np.delete( all_mage, tx, axis=0)
    np.delete( all_col, tx, axis=0)
    np.delete( all_cole, tx, axis=0)
    del all_Nc,tx

    fout = open( output_catalog_filename, 'w' )
    if number_of_test_galaxies == None:
        for i in range(len(all_id)):
            results = get_photoz( all_col[i,:], all_cole[i,:], all_id[i], all_col, all_z, all_id, use_thresh_table, \
                minimum_Ncolors=minimum_number_of_colors, ppf_value=use_ppf_value )
            fout.write( '%i %6.4f %6.4f %6.4f %i \n' % ( all_id[i], all_z[i], results[0], results[1], results[2] ) )
    else:
        tx = np.random.choice( len(all_id), size=number_of_test_galaxies, replace=False )
        for i in tx:
            results = get_photoz( all_col[i,:], all_cole[i,:], all_id[i], all_col, all_z, all_id, use_thresh_table, \
                minimum_Ncolors=minimum_number_of_colors, ppf_value=use_ppf_value )
            fout.write( '%i %6.4f %6.4f %6.4f %i \n' % ( all_id[i], all_z[i], results[0], results[1], results[2] ) )
        del tx
    fout.close()


if __name__ == '__main__':
    run( 'cat.dat', 'zphot.dat', 6, number_of_test_galaxies=20000 )






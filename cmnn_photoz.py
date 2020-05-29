import numpy as np
from scipy.stats import chi2
import datetime


def return_photoz( test_c, test_ce, test_id, train_c, train_z, train_id, \
    ppf_value, thresh_table, selection_mode, minimum_Ncolors, minimum_Nneighbors ):

    ### For a single test galaxy, return photometric redshift and uncertainty based
    ###  on the supplied training-set galaxies and CMNN Estimator mode parameters.

    ### Inputs
    ###   test_c             : array of colors for test galaxy
    ###   test_ce            : array of color errors for test galaxy
    ###   test_id            : unique integer identifier for test galaxy
    ###   train_c            : array of colors for all training-set galaxies
    ###   train_z            : array of color errors for all training-set galaxies
    ###   train_id           : array of unique integer identifiers for all training-set galaxies
    ###   ppf_value          : percent point function value (typically 0.68 or 0.95)
    ###   thresh_table       : table of thresholds to apply based on the ppf_value
    ###   selection_mode     : how the photo-z will be selected from the CMNN subset of training galaxies
    ###                          0 : random, 1 : nearest neighbor, 2 : weighted random
    ###   minimum_Ncolors    : minimum number of colors used to identify the CMNN subset of training galaxies
    ###   minimum_Nneighbors : the minimum size of the CMNN subset of training galaxies

    ### Outputs returned
    ###   Photoz      : the photometric redshift for the test galaxy
    ###   PhotozError : the uncertainty in the photo-z for the test galaxy
    ###   NMC         : the number of training-set galaxies in the color-matched subset

    ### Calculate the Mahalanobis Distance and the number of degrees of freedom for each training set galaxy
    MahalanobisDistance = np.nansum( ( test_c - train_c )**2 / test_ce**2, axis=1 )
    DegreesOfFreedom    = np.nansum( ( test_c - train_c ) / ( test_c - train_c ), axis=1 )

    ### Determine the appropriate threshold that should apply to each training set galaxy
    ###   Could do the slow way:  thresholds = chi2.ppf( ppf_value, DegreesOfFreedom )
    thresholds = np.zeros( len(train_c), dtype='float' )
    ### Only calculate the thresholds for training galaxies that could conceivably be a CMNN
    tx = np.where( ( np.isfinite(MahalanobisDistance) ) & ( np.isfinite(DegreesOfFreedom) ) &\
     (DegreesOfFreedom >= minimum_Ncolors) & (MahalanobisDistance <= thresh_table[-1]) )[0]
    if len(tx) >= 1:
        for i in tx:
            thresholds[i] = thresh_table[ DegreesOfFreedom[i] ]
    del tx

    ### Identify the indicies of the CMNN subset of training-set galaxies
    index = np.where( \
        ( test_id != train_id ) & \
        ( np.isfinite(MahalanobisDistance) ) & \
        ( np.isfinite(DegreesOfFreedom) ) & \
        ( DegreesOfFreedom >= minimum_Ncolors ) & \
        ( thresholds > 0.0010 ) & \
        ( MahalanobisDistance > 0.00010 ) & \
        ( MahalanobisDistance < thresholds ) )[0]

    ### Determine the photometric redshift for this test galaxy
    ### if there are a sufficient number of training-set galaxies in the CMNN subset
    if len(index) >= minimum_Nneighbors:
        ### choose randomly from the color matched sample
        if selection_mode == 0:
            rval  = int(np.random.uniform(low=0, high=len(index)-1))
            rival = index[rval]
            Photoz      = train_z[rival]
            PhotozError = np.std( train_z[index] )
            del rval,rival
        ### choose the nearest neighbor, the best color match
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
        ### weight by how good the color match is and then choose randomly
        if selection_mode == 2:
            tweight = float(1.00) / MahalanobisDistance[index]
            weight  = tweight / np.sum(tweight)
            rval    = np.random.choice( range(len(index)), size=1, replace=False, p=weight )
            rival   = index[rval]
            Photoz      = train_z[rival]
            PhotozError = np.std( train_z[index] )
            del tweight,weight,rval,rival
        NMC = len(index)

    ### if there are too few training-set galaxies in the CMNN subset
    if len(index) < minimum_Nneighbors:
        ### find out how many there are we could potentially use
        index2 = np.where( ( np.isfinite(MahalanobisDistance) ) & ( np.isfinite(DegreesOfFreedom) ) & \
            (MahalanobisDistance > 0.0) & (DegreesOfFreedom >= minimum_Ncolors) )[0]

        ### if there's more than the minimum number, use them
        if len(index2) >= minimum_Nneighbors:
            tempMD = MahalanobisDistance[index2]
            tempTZ = train_z[index2]
            tempDF = DegreesOfFreedom[index2]
            ### identify the nearest neighbors and use them as the CMNN subset
            sx = np.argsort( tempMD )
            new_MD = np.asarray( tempMD[sx[0:minimum_Nneighbors]], dtype='float' )
            new_TZ = np.asarray( tempTZ[sx[0:minimum_Nneighbors]], dtype='float' )
            new_DF = np.asarray( tempDF[sx[0:minimum_Nneighbors]], dtype='float' )
            del tempMD,tempTZ,tempDF,sx
            ### calculate the new 'effective PPF' based on the most distant nearest neighbor
            # new_ppf_value = chi2.cdf( new_MD[minimum_Nneighbors-1], new_DF[minimum_Nneighbors-1] )
            new_ppf_value = chi2.cdf( new_MD[-1], new_DF[-1] )
            ### inflate the photo-z error appropriately
            temp   = np.std( new_TZ )
            PhotozError = temp * (new_ppf_value / ppf_value)
            del temp,new_ppf_value
            ### choose randomly from nearest dselect galaxies
            if selection_mode == 0:
                rval   = int( np.floor( np.random.uniform(low=0, high=len(new_TZ)-1) ) )
                Photoz = new_TZ[rval]
                del rval
            ### choose nearest neighbour, use nearest dselect for error
            if selection_mode == 1:
                Photoz = new_TZ[0]
            ### weight by how good the color match is and then select
            if selection_mode == 2:
                tweight = float(1.00) / new_MD
                weight = tweight / np.sum(tweight)
                cx     = np.random.choice( range(len(new_TZ)), size=1, replace=False, p=weight )
                Photoz = new_TZ[cx]
                del tweight,weight,cx
            del new_MD,new_TZ,new_DF
            ### set the number in the CMNN subset to be minimum_Nneighbors
            NMC = minimum_Nneighbors

        ### if there's not enough training-set galaxies this is probably a bad test galaxy anway
        else:
            Photoz = -99.99
            PhotozError = -99.99
            NMC = 0

        del index2

    del index, MahalanobisDistance, DegreesOfFreedom, thresholds 

    return [Photoz, PhotozError, NMC]


def make_zphot(verbose, runid, force_idet, cmnn_minNc, cmnn_minNN, cmnn_ppf, cmnn_rsel, cmnn_ppmag, cmnn_ppclr):
    if verbose:
        print(' ')
        print('Starting cmnn_photoz.make_zphot(), ',datetime.datetime.now())

    if verbose: print('Reading test and train catalogs in output/run_'+runid+'/')
    all_test_id = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='int', usecols={0} )
    all_test_tz = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={1} )
    all_test_m  = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={2,4,6,8,10,12} )
    # all_test_me = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={3,5,7,9,11,13} )
    all_test_c  = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={14,16,18,20,22} )
    all_test_ce = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={15,17,19,21,23} )

    all_train_id = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='int', usecols={0} )
    all_train_tz = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={1} )
    all_train_m  = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={2,4,6,8,10,12} )
    # all_train_me = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={3,5,7,9,11,13} )
    all_train_c  = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={14,16,18,20,22} )
    # all_train_ce = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={15,17,19,21,23} )

    ### Prepare a table of thresholds based on the desired percent point function
    ###   chi2.ppf is slow, so we only want to do this once
    cmnn_thresh_table = np.zeros( 6, dtype='float' )
    for i in range(6):
        cmnn_thresh_table[i] = chi2.ppf(cmnn_ppf,i)

    ### Prepare for a magnitude pre-cut on the training set
    if (cmnn_ppmag == True) & (force_idet == False):
        print('Error. Must set force_idet = True in order to set the cmnn_ppmag = True.')
        print('  cmnn_ppmag : %r \n' % cmnn_ppmag)
        print('  force_idet : %r \n' % force_idet)
        print('Exit (cmnn_ppmag and force_idet user inputs are incompatible).')
        exit()
    if (cmnn_ppmag == True) & (force_idet == True):
        ppmag_sorted_train_imags = np.sort( all_train_m[:,3] )
        ppmag_fractions = np.asarray( range(len(ppmag_sorted_train_imags)), dtype='float')/float(len(ppmag_sorted_train_imags))


    ### Calculate photometric redshifts for all test-set galaxies and write to zphot file
    if verbose: print('Starting to create list of photo-z: output/run_'+runid+'/zphot.cat')
    fout = open('output/run_'+runid+'/zphot.cat','w')
    for i in range(len(all_test_id)):
        ### if cmnn_ppmag or cmnn_ppclr is true, we only use part of the training set
        if cmnn_ppmag or cmnn_ppclr:
            ### set default values that are equivalent to 'no cut'
            ilow  = 15.0
            ihi   = 30.0
            grlow = -10.0
            grhi  = +10.0
            rilow = -10.0
            rihi  = +10.0
            ### define the lower/upper i-band magnitudes for the training set
            if cmnn_ppmag:
                mx = np.argmin( np.abs( all_test_m[i,3] - ppmag_sorted_train_imags ) )
                ### percentile of the test-set galaxy's i-band magnitude
                pc = ppmag_fractions[mx]
                ### find i-band mags bounding +/-5% of training galaxies
                pclow = pc - 0.05
                pchi  = pc + 0.05
                if pclow < 0.00:
                    pclow = 0.00
                    pchi  = 0.10
                if pchi > 1.00:
                    pclow = 0.90
                    pchi  = 1.00
                pxlow = np.argmin( np.abs( pclow - ppmag_fractions ) )
                pxhi  = np.argmin( np.abs( pchi - ppmag_fractions ) )
                ilow = ppmag_sorted_train_imags[pxlow]
                ihi  = ppmag_sorted_train_imags[pxhi]
                del mx,pc,pclow,pchi,pxlow,pxhi
            ### define lower/upper g-r and r-i colors for the training set
            if cmnn_ppclr:
                if (np.isfinite(all_test_c[i,1])) & (np.isfinite(all_test_c[i,2])):
                    grlow = all_test_c[i,1]-0.3
                    grhi  = all_test_c[i,1]+0.3
                    rilow = all_test_c[i,2]-0.3
                    rihi  = all_test_c[i,2]+0.3
            ### apply the boundaries in i mag and g-r, r-i color to the training set
            trx = np.where( (all_train_m[:,3] >= ilow) & (all_train_m[:,3] <= ihi) &\
                (all_train_c[:,1] >= grlow) & (all_train_c[:,1] <= grhi) &\
                (all_train_c[:,2] >= rilow) & (all_train_c[:,2] <= rihi) )[0]
            ### now get the photoz for this test galaxy
            results = return_photoz( all_test_c[i], all_test_ce[i], all_test_id[i], \
                all_train_c[trx], all_train_tz[trx], all_train_id[trx], \
                cmnn_ppf, cmnn_thresh_table, cmnn_rsel, cmnn_minNc, cmnn_minNN)
            del trx,ilow,ihi,grlow,grhi,rilow,rihi
        ### if cmnn_ppmag and cmnn_ppclr are false, we use the whole training set
        else:
            results = return_photoz( all_test_c[i], all_test_ce[i], all_test_id[i], \
                all_train_c, all_train_tz, all_train_id, \
                cmnn_ppf, cmnn_thresh_table, cmnn_rsel, cmnn_minNc, cmnn_minNN)       
        fout.write( '%10i %6.4f %6.4f %6.4f %10i \n' % (all_test_id[i], all_test_tz[i], results[0], results[1], results[2]) )
    fout.close()
    if verbose: print('Wrote to: output/run_'+runid+'/zphot.cat')




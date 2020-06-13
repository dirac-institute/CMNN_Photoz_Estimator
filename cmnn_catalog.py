import os
import numpy as np
import matplotlib.pyplot as plt
import datetime


def make_test_and_train(verbose, runid, test_m5, train_m5, test_mcut, train_mcut, force_idet, \
    test_N, train_N, cmnn_minNc):

    ### Make test and training set catalogs based on user input.
    ### All inputs are described in cmnn_run.py, and they are assumed to
    ###  have been vetted by cmnn_run.py, and are not rechecked here.

    if verbose:
        print(' ')
        print('Starting cmnn_catalog.make_test_and_train(), ',datetime.datetime.now())

    ### Read in the raw catalog of galaxies
    if verbose: print('Read the mock catalog of true redshifts and magnitudes.')

    ### This commented-out code can be used to speed up the catalog read, under some conditions,
    ###    BUT ONLY WHEN the FULL catalog is being used. But that is not the default.
    # if force_idet:
    #     ### Speed up this read by dropping galaxies we don't need:
    #     ###  any galaxy that's more than half a mag fainter than the faintest i-band cut
    #     if verbose: print('Speed things up using awk to pre-select useful galaxies from big data file.')
    #     imagmax = np.max( [ test_mcut[3], train_mcut[3] ] ) + 0.5
    #     strimagmax = str(np.round(imagmax,2))
    #     if verbose: print("awk '{if ($5<"+strimagmax+") print $0}' LSST_galaxy_catalog_full.dat > temp.dat")
    #     os.system("awk '{if ($5<"+strimagmax+") print $0}' LSST_galaxy_catalog_full.dat > temp.dat")
    #     all_id = np.loadtxt( 'temp.dat', dtype='float', usecols={0})
    #     all_tz = np.loadtxt( 'temp.dat', dtype='float', usecols={1})
    #     all_tm = np.loadtxt( 'temp.dat', dtype='float', usecols={2,3,4,5,6,7})
    #     if verbose: print('rm temp.dat')
    #     os.system('rm temp.dat')
    # else:
    #     all_id = np.loadtxt( 'LSST_galaxy_catalog_full.dat', dtype='float', usecols={0})
    #     all_tz = np.loadtxt( 'LSST_galaxy_catalog_full.dat', dtype='float', usecols={1})
    #     all_tm = np.loadtxt( 'LSST_galaxy_catalog_full.dat', dtype='float', usecols={2,3,4,5,6,7})

    ### Check if the gzip needs unzipping
    if (os.path.isfile( 'LSST_galaxy_catalog_i25p3.dat' ) == False) & \
       (os.path.isfile( 'LSST_galaxy_catalog_i25p3.dat.gz' ) == True):
        os.system('gunzip LSST_galaxy_catalog_i25p3.dat.gz')

    if (os.path.isfile( 'LSST_galaxy_catalog_i25p3.dat' ) == False) & \
       (os.path.isfile( 'LSST_galaxy_catalog_i25p3.dat.gz' ) == False):
        print('Error. Mock galaxy catalog file is missing or misnamed.')
        print('Required to have one of the following:')
        print('  LSST_galaxy_catalog_i25p3.dat')
        print('  LSST_galaxy_catalog_i25p3.dat.gz')
        print('Exit (missing input file).')
        exit()

    all_id = np.loadtxt( 'LSST_galaxy_catalog_i25p3.dat', dtype='float', usecols={0})
    all_tz = np.loadtxt( 'LSST_galaxy_catalog_i25p3.dat', dtype='float', usecols={1})
    all_tm = np.loadtxt( 'LSST_galaxy_catalog_i25p3.dat', dtype='float', usecols={2,3,4,5,6,7})

    ### Ensure needed quantities are in numpy arrays
    gamma = np.asarray( [0.037,0.038,0.039,0.039,0.04,0.04], dtype='float' )
    np_test_m5    = np.asarray( test_m5, dtype='float' )
    np_train_m5   = np.asarray( train_m5, dtype='float' )
    np_test_mcut  = np.asarray( test_mcut, dtype='float' )
    np_train_mcut = np.asarray( train_mcut, dtype='float' )

    ### Calculate the magnitude errors based on the m5 depths
    if verbose: print('Calculating magnitude errors.')
    all_test_me = np.sqrt( ( 0.04 - gamma ) * ( np.power( 10.0, 0.4*(all_tm[:]-np_test_m5) ) ) + \
        gamma * ( np.power( 10.0, 0.4*(all_tm[:]-np_test_m5) )**2 ) )
    all_train_me = np.sqrt( ( 0.04 - gamma ) * ( np.power( 10.0, 0.4*(all_tm[:]-np_train_m5) ) ) + \
        gamma * ( np.power( 10.0, 0.4*(all_tm[:]-np_train_m5) )**2 ) )

    ### Set error floor
    for f in range(6):
        tex = np.where( all_test_me[:,f] < 0.0050)[0]
        all_test_me[tex,f] = float(0.0050)
        trx = np.where( all_train_me[:,f] < 0.0050)[0]
        all_train_me[trx,f] = float(0.0050)

    ### Calculate observed apparent magnitudes based on the errors
    if verbose: print('Calculating observed apparent magnitudes.')
    all_test_m = all_tm + all_test_me * np.random.normal( size = (len(all_tm),6) )
    all_train_m = all_tm + all_train_me * np.random.normal( size = (len(all_tm),6) )

    ### Do not allow tm < 18, the approximate saturation point
    for f in range(6):
        tx = np.where( all_tm[:,f] < 18.0000 )[0]
        all_test_me[tx]  = float('NaN')
        all_test_m[tx]   = float('NaN')
        all_train_me[tx] = float('NaN')
        all_train_m[tx]  = float('NaN')
        del tx

    ### Do not allow (tm-m5) > 0.2.
    ###  For tm = m5, then me = 0.2, and allowing tm-m5>0.5 allows "upscattering"
    ###  of faint galaxies to brighter observed apparent magnitude, getting them "detected"
    for f in range(6):
        tx = np.where( all_tm[:,f] > np_test_m5[f]+0.2000 )[0]
        all_test_me[tx] = float('NaN')
        all_test_m[tx] = float('NaN')
        del tx
        tx = np.where( all_tm[:,f] > np_train_m5[f]+0.2000 )[0]
        all_train_me[tx] = float('NaN')
        all_train_m[tx] = float('NaN')
        del tx

    ### Apply the magnitude cuts, set values to float('NaN') if not detected
    ### If the user desired to force a detection in i-band, then for all galaxies
    ###   undetected in i-band, set all filters' apparent mags to float('NaN')
    if verbose: print('Applying the magnitude cuts.')
    for f in range(6):
        te_x = np.where( all_test_m[:,f] > np_test_mcut[f] )[0]
        if len(te_x) > 0:
            all_test_m[te_x,f]  = float('NaN')
            all_test_me[te_x,f] = float('NaN')
        if (force_idet == True) & (f == 3):
            all_test_m[te_x,:]  = float('NaN')
            all_test_me[te_x,:] = float('NaN')
        tr_x = np.where( all_train_m[:,f] > np_train_mcut[f] )[0]
        if len(tr_x) > 0:
            all_train_m[tr_x,f]  = float('NaN')
            all_train_me[tr_x,f] = float('NaN')
        if (force_idet == True) & (f == 3):
            all_train_m[tr_x,:]  = float('NaN')
            all_train_me[tr_x,:] = float('NaN')
        del te_x,tr_x

    ### Calculate colors, color errors, and number of colors for each galaxy
    if verbose: print('Calculating colors.')
    all_test_c = np.zeros( (len(all_tm),5), dtype='float' )
    all_test_ce = np.zeros( (len(all_tm),5), dtype='float' )
    all_train_c = np.zeros( (len(all_tm),5), dtype='float' )
    all_train_ce = np.zeros( (len(all_tm),5), dtype='float' )
    for c in range(5):
        all_test_c[:,c]   = all_test_m[:,c] - all_test_m[:,c+1]
        all_train_c[:,c]  = all_train_m[:,c] - all_train_m[:,c+1]
        all_test_ce[:,c]  = np.sqrt( all_test_me[:,c]**2  + all_test_me[:,c+1]**2 )
        all_train_ce[:,c] = np.sqrt( all_train_me[:,c]**2 + all_train_me[:,c+1]**2 )
    all_test_Nc  = np.nansum( all_test_c/all_test_c, axis=1 )
    all_train_Nc = np.nansum( all_train_c/all_train_c, axis=1 )

    ### The number of potential test and training set galaxies must be greater than desired catalog size
    te_x = np.where( all_test_Nc >= cmnn_minNc )[0]
    tr_x = np.where( all_train_Nc >= cmnn_minNc )[0]
    if (len(te_x) < test_N) | (len(tr_x) < train_N):
        print('Error. Desired number of test/train galaxies higher than what is available.')
        print('  test number desired, available: %i %i' % (test_N,len(te_x)) )
        print('  train number desired, available: %i %i' % (train_N,len(tr_x)) )
        print('Exit (inputs too constraining to build test/train set).')
        exit()
    else:
        ### Create test.cat
        if verbose: print('Opening and writing to ','output/run_'+runid+'/test.cat')
        te_rx = np.random.choice( te_x, size=test_N, replace=False )
        test_fout = open('output/run_'+runid+'/test.cat','w')
        for i in te_rx:
            test_fout.write(' %10i %10.8f ' % (all_id[i],all_tz[i]) )
            for f in range(6):
                test_fout.write('%9.6f %9.6f ' % (all_test_m[i,f],all_test_me[i,f]) )
            for c in range(5):
                test_fout.write('%9.6f %9.6f ' % (all_test_c[i,c],all_test_ce[i,c]) )
            test_fout.write('\n')
        test_fout.close()
        del te_rx,test_fout
        ### Create train.cat
        if verbose: print('Opening and writing to ','output/run_'+runid+'/train.cat')
        tr_rx = np.random.choice( tr_x, size=train_N, replace=False )
        train_fout = open('output/run_'+runid+'/train.cat','w')
        for i in tr_rx:
            train_fout.write(' %10i %10.8f ' % (all_id[i],all_tz[i]) )
            for f in range(6):
                train_fout.write('%9.6f %9.6f ' % (all_train_m[i,f],all_train_me[i,f]) )
            for c in range(5):
                train_fout.write('%9.6f %9.6f ' % (all_train_c[i,c],all_train_ce[i,c]) )
            train_fout.write('\n')
        train_fout.close()
        del tr_rx,train_fout

        if verbose: print('Wrote: ','output/run_'+runid+'/test.cat, output/run_'+runid+'/train.cat')


def make_plots(verbose, runid):

    ### Make redshift and magnitude histograms for the test and training sets.

    if verbose:
        print(' ')
        print('Starting cmnn_catalog.make_plots(), ',datetime.datetime.now())
    
    if verbose: print('Reading test and train catalogs in output/run_'+runid+'/')

    if os.path.isdir('output/run_'+runid+'/plot_cats') == False :
        os.system('mkdir output/run_'+runid+'/plot_cats')

    fnm = 'output/run_'+runid+'/test.cat'
    test_tz = np.loadtxt( fnm, dtype='float', usecols={1})
    test_m  = np.loadtxt( fnm, dtype='float', usecols={2,4,6,8,10,12})
    test_me = np.loadtxt( fnm, dtype='float', usecols={3,5,7,9,11,13})
    # test_c  = np.loadtxt( fnm, dtype='float', usecols={14,16,18,20,22})
    # test_ce = np.loadtxt( fnm, dtype='float', usecols={15,17,19,21,23})

    fnm = 'output/run_'+runid+'/train.cat'
    train_tz = np.loadtxt( fnm, dtype='float', usecols={1})
    train_m  = np.loadtxt( fnm, dtype='float', usecols={2,4,6,8,10,12})
    train_me = np.loadtxt( fnm, dtype='float', usecols={3,5,7,9,11,13})
    # train_c  = np.loadtxt( fnm, dtype='float', usecols={14,16,18,20,22})
    # train_ce = np.loadtxt( fnm, dtype='float', usecols={15,17,19,21,23})


    ### Histogram of true redshift for train and test
    pfnm = 'output/run_'+runid+'/plot_cats/hist_ztrue'
    fig  = plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size':20})
    plt.hist( test_tz,  normed=True,bins=30,histtype='step',ls='solid',lw=2,alpha=1,  \
        color='black',label='test')
    plt.hist( train_tz, normed=True,bins=30,histtype='step',ls='solid',lw=4,alpha=0.5,\
        color='black',label='train')
    plt.xlabel('True Catalog Redshift')
    plt.ylabel('Fraction of Galaxies')
    plt.legend(loc='upper right', prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm,bbox_inches='tight')
    if verbose: print('Made '+pfnm)

    ### Histograms of observed apparent magnitude
    pfnm = 'output/run_'+runid+'/plot_cats/hist_mag'
    fig  = plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size':20})
    filt_names  = ['u','g','r','i','z','y']
    filt_colors = ['darkviolet','darkgreen','red','darkorange','brown','black']
    for f in range(6):
        tex = np.where( np.isfinite(test_m[:,f]) )[0]
        plt.hist( test_m[tex,f], normed=True,cumulative=True,bins=30,histtype='step',ls='solid',\
            lw=2,alpha=1,  \
            color=filt_colors[f],label='test '+filt_names[f])
        trx = np.where( np.isfinite(train_m[:,f]) )[0]
        plt.hist( train_m[trx,f], normed=True,cumulative=True,bins=30,histtype='step',ls='solid',\
            lw=4,alpha=0.5,  \
            color=filt_colors[f],label='train '+filt_names[f])
        del tex,trx
    plt.xlabel('Observed Apparent Magnitude')
    plt.ylabel('Fraction of Galaxies')
    plt.legend(loc='upper left', prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm,bbox_inches='tight')
    if verbose: print('Made '+pfnm)

    ### Histograms of observed apparent magnitude error
    mebins = np.arange( 26, dtype='float' )*0.01
    pfnm = 'output/run_'+runid+'/plot_cats/hist_mage'
    fig  = plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size':20})
    filt_names  = ['u','g','r','i','z','y']
    filt_colors = ['darkviolet','darkgreen','red','darkorange','brown','black']
    for f in range(6):
        tex = np.where( np.isfinite(test_m[:,f]) )[0]
        plt.hist( test_me[tex,f], normed=True,bins=mebins,histtype='step',ls='solid',\
            lw=2,alpha=1,color=filt_colors[f],label='test '+filt_names[f])
        trx = np.where( np.isfinite(train_m[:,f]) )[0]
        plt.hist( train_me[trx,f], normed=True,bins=mebins,histtype='step',ls='solid',\
            lw=4,alpha=0.5,color=filt_colors[f],label='train '+filt_names[f])
        del tex,trx
    plt.xlabel('Apparent Magnitude Error')
    plt.ylabel('Fraction of Galaxies')
    plt.legend(loc='upper right', prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm,bbox_inches='tight')
    if verbose: print('Made '+pfnm)

    ### Magnitude error vs. observed apparent magnitude
    ###  Just use 10000 random test galaxies
    pfnm = 'output/run_'+runid+'/plot_cats/test_mag_vs_mage'
    fig  = plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size':20})
    filt_names  = ['u','g','r','i','z','y']
    filt_colors = ['darkviolet','darkgreen','red','darkorange','brown','black']
    for f in range(6):
        tex = np.where( np.isfinite(test_m[:,f]) )[0]
        tx = np.random.choice( tex, size=10000, replace=False )
        plt.plot( test_m[tx,f], test_me[tx,f], 'o',alpha=0.5,mew=0, \
            color=filt_colors[f],label=filt_names[f])
        del tex,tx
    plt.xlabel('Observed Apparent Magnitude')
    plt.ylabel('Apparent Magnitude Error')
    plt.legend(loc='upper left', prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm,bbox_inches='tight')
    if verbose: print('Made '+pfnm)

    pfnm = 'output/run_'+runid+'/plot_cats/train_mag_vs_mage'
    fig  = plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size':20})
    filt_names  = ['u','g','r','i','z','y']
    filt_colors = ['darkviolet','darkgreen','red','darkorange','brown','black']
    for f in range(6):
        trx = np.where( np.isfinite(train_m[:,f]) )[0]
        tx = np.random.choice( trx, size=10000, replace=False )
        plt.plot( train_m[tx,f], train_me[tx,f], 'o',alpha=0.5,mew=0, \
            color=filt_colors[f],label=filt_names[f])
        del trx,tx
    plt.xlabel('Observed Apparent Magnitude')
    plt.ylabel('Apparent Magnitude Error')
    plt.legend(loc='upper left', prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm,bbox_inches='tight')
    if verbose: print('Made '+pfnm)


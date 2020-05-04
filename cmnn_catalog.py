import os
import numpy as np
import matplotlib.pyplot as plt
import datetime


def make_test_and_train(verbose, runid, test_m5, train_m5, test_mcut, train_mcut, force_idet, test_N, train_N, cmnn_minNc):

    ### Make test and training set catalogs based on user input.
    ### All inputs are described in cmnn_run.py, and they are assumed to
    ###  have been vetted by cmnn_run.py, and are not rechecked here.

    if verbose:
        print(' ')
        print('Starting cmnn_catalog.make_test_and_train(), ',datetime.datetime.now())

    ### Read in the raw catalog of galaxies
    if verbose: print('Read the raw catalog.')
    if force_idet:
        ### Speed up this read by dropping galaxies we don't need:
        ###  any galaxy that's more than half a mag fainter than the faintest i-band cut
        if verbose: print('Speed things up using awk to pre-select useful galaxies from big data file.')
        imagmax = np.max( [ test_mcut[3], train_mcut[3] ] ) + 0.5
        strimagmax = str(np.round(imagmax,2))
        if verbose: print("awk '{if ($5<"+strimagmax+") print $0}' LSST_galaxy_catalog_full.dat > temp.dat")
        os.system("awk '{if ($5<"+strimagmax+") print $0}' LSST_galaxy_catalog_full.dat > temp.dat")
        all_id = np.loadtxt( 'temp.dat', dtype='float', usecols={0})
        all_tz = np.loadtxt( 'temp.dat', dtype='float', usecols={1})
        all_tm = np.loadtxt( 'temp.dat', dtype='float', usecols={2,3,4,5,6,7})
        if verbose: print('rm temp.dat')
        os.system('rm temp.dat')
    else:
        all_id = np.loadtxt( 'LSST_galaxy_catalog_full.dat', dtype='float', usecols={0})
        all_tz = np.loadtxt( 'LSST_galaxy_catalog_full.dat', dtype='float', usecols={1})
        all_tm = np.loadtxt( 'LSST_galaxy_catalog_full.dat', dtype='float', usecols={2,3,4,5,6,7})

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
            test_fout.write(' %10i %8.6f ' % (all_id[i],all_tz[i]) )
            for f in range(6):
                test_fout.write('%6.3f %6.3f ' % (all_test_m[i,f],all_test_me[i,f]) )
            for c in range(5):
                test_fout.write('%6.3f %6.3f ' % (all_test_c[i,c],all_test_ce[i,c]) )
            test_fout.write('\n')
        test_fout.close()
        del te_rx,test_fout
        ### Create train.cat
        if verbose: print('Opening and writing to ','output/run_'+runid+'/train.cat')
        tr_rx = np.random.choice( tr_x, size=train_N, replace=False )
        train_fout = open('output/run_'+runid+'/train.cat','w')
        for i in tr_rx:
            train_fout.write(' %10i %8.6f ' % (all_id[i],all_tz[i]) )
            for f in range(6):
                train_fout.write('%6.3f %6.3f ' % (all_train_m[i,f],all_train_me[i,f]) )
            for c in range(5):
                train_fout.write('%6.3f %6.3f ' % (all_train_c[i,c],all_train_ce[i,c]) )
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
    # test_id = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={0})
    test_tz = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={1})
    test_m  = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={2,4,6,8,10,12})
    # test_me = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={3,5,7,9,11,13})
    # test_c  = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={14,16,18,20,22})
    # test_ce = np.loadtxt( 'output/run_'+runid+'/test.cat', dtype='float', usecols={15,17,19,21,23})

    # train_id = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={0})
    train_tz = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={1})
    train_m  = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={2,4,6,8,10,12})
    # train_me = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={3,5,7,9,11,13})
    # train_c  = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={14,16,18,20,22})
    # train_ce = np.loadtxt( 'output/run_'+runid+'/train.cat', dtype='float', usecols={15,17,19,21,23})

    pfnm = 'output/run_'+runid+'/cat_hist_ztrue'
    fig  = plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size':20})
    zbw   = float(0.1)
    zbins = np.arange( 36, dtype='float' )*zbw
    test_y = np.zeros( 36, dtype='float' )
    train_y = np.zeros( 36, dtype='float' )
    for z,zbin in enumerate(zbins):
        if z > 0:
            tex = np.where( (test_tz  > zbins[z-1]) & (test_tz  <= zbins[z] ) )[0]
            trx = np.where( (train_tz > zbins[z-1]) & (train_tz <= zbins[z] ) )[0]
            test_y[z]  = float(len(tex))/float(len(test_tz))
            train_y[z] = float(len(trx))/float(len(train_tz))
            del tex,trx
            plt.plot( [zbins[z-1],zbins[z-1],zbins[z]], [test_y[z-1],test_y[z],test_y[z]], \
                ls='solid',lw=2,alpha=1,color='black' )
            plt.plot( [zbins[z-1],zbins[z-1],zbins[z]], [train_y[z-1],train_y[z],train_y[z]], \
                ls='solid',lw=4,alpha=0.5,color='black' )
    plt.plot( [zbins[-1],zbins[-1]], [test_y[z],0.0], ls='solid',lw=2,alpha=1,color='black',label='test')
    plt.plot( [zbins[-1],zbins[-1]], [train_y[z],0.0], ls='solid',lw=4,alpha=0.5,color='black',label='train')
    del zbw,zbins,test_y,train_y
    plt.xlabel('True Catalog Redshift')
    plt.ylabel('Fraction of Galaxies')
    plt.legend(loc='upper right', prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm,bbox_inches='tight')
    if verbose: print('Made '+pfnm)

    pfnm = 'output/run_'+runid+'/cat_hist_mag'
    fig  = plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size':20})
    filt_names  = ['u','g','r','i','z','y']
    filt_colors = ['darkviolet','darkgreen','red','darkorange','brown','black']
    mbw = float(0.25)
    mbins = np.arange( 41, dtype='float' )*mbw + 18.0
    for f in range(6):
        tex = np.where( np.isfinite(test_m[:,f]) )[0]
        trx = np.where( np.isfinite(train_m[:,f]) )[0]
        Ntest = float(len(tex))
        Ntrain = float(len(trx))
        del tex,trx
        test_y = np.zeros( 41, dtype='float' )
        train_y = np.zeros( 41, dtype='float' )
        for m,mbin in enumerate(mbins):
            if m > 0:
                tex = np.where( (test_m[:,f]  > mbins[m-1]) & (test_m[:,f] <= mbins[m])  & (np.isfinite(test_m[:,f])) )[0]
                trx = np.where( (train_m[:,f] > mbins[m-1]) & (train_m[:,f] <= mbins[m]) & (np.isfinite(train_m[:,f])) )[0]
                test_y[m] = float(len(tex)) / Ntest
                train_y[m] = float(len(trx)) / Ntrain
                del tex,trx
                plt.plot( [mbins[m-1],mbins[m-1],mbins[m]], [test_y[m-1],test_y[m],test_y[m]], \
                    ls='solid',lw=2,alpha=1,color=filt_colors[f] )
                plt.plot( [mbins[m-1],mbins[m-1],mbins[m]], [train_y[m-1],train_y[m],train_y[m]], \
                    ls='solid',lw=4,alpha=0.5,color=filt_colors[f] )
        plt.plot( [mbins[-1],mbins[-1]], [test_y[m],0.0], ls='solid',lw=2,alpha=1,color=filt_colors[f],label=filt_names[f]+' test')
        plt.plot( [mbins[-1],mbins[-1]], [train_y[m],0.0], ls='solid',lw=4,alpha=0.5,color=filt_colors[f],label=filt_names[f]+' train')
        del test_y,train_y,Ntest,Ntrain
    del mbw,mbins
    plt.xlabel('Apparent Magnitude')
    plt.ylabel('Fraction of Galaxies')
    plt.legend(loc='upper left', prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm,bbox_inches='tight')
    if verbose: print('Made '+pfnm)



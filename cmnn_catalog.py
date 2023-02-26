import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

def make_test_and_train(verbose, runid, filtmask, yfilt, catalog, 
                        test_m5, train_m5, test_mcut, train_mcut, 
                        force_idet, force_gridet, test_N, train_N, cmnn_minNc):
    
    if verbose:
        print('Starting cmnn_catalog.make_test_and_train: ',datetime.datetime.now())

    all_id = np.loadtxt(catalog, dtype='float', usecols=(0))
    all_tz = np.loadtxt(catalog, dtype='float', usecols=(1))
    
    # yfilt = 0 : use PanSTARRs y-band (default, column 7)
    # yfilt = 1 : use Euclid y-band (column 8)
    if yfilt == 0:
        all_tm = np.loadtxt(catalog, dtype='float', usecols=(2,3,4,5,6,7,9,10,11))
    elif yfilt == 1:
        all_tm = np.loadtxt(catalog, dtype='float', usecols=(2,3,4,5,6,8,9,10,11))
        
    # gamma sets the impact of sky brightness in magnitude error estimates
    gamma = np.asarray( [0.037,0.038,0.039,0.039,0.04,0.04,0.04,0.04,0.04], dtype='float' )

    np_test_m5    = np.asarray( test_m5, dtype='float' )
    np_train_m5   = np.asarray( train_m5, dtype='float' )
    np_test_mcut  = np.asarray( test_mcut, dtype='float' )
    np_train_mcut = np.asarray( train_mcut, dtype='float' )

    all_test_me = np.sqrt((0.04 - gamma) * (np.power(10.0, 0.4*(all_tm[:]-np_test_m5))) + \
                          gamma * (np.power(10.0, 0.4*(all_tm[:]-np_test_m5))**2))
    all_train_me = np.sqrt((0.04 - gamma) * (np.power(10.0, 0.4*(all_tm[:]-np_train_m5))) + \
                           gamma * (np.power(10.0, 0.4*(all_tm[:]-np_train_m5))**2))
    
    # apply the uncertainty floor of 0.005 mag
    for f in range(9):
        tex = np.where( all_test_me[:,f] < 0.0050)[0]
        all_test_me[tex,f] = float(0.0050)
        trx = np.where( all_train_me[:,f] < 0.0050)[0]
        all_train_me[trx,f] = float(0.0050)
    
    # use the errors to calculate apparent observed magnitudes
    all_test_m = all_tm + all_test_me * np.random.normal(size=(len(all_tm), 9))
    all_train_m = all_tm + all_train_me * np.random.normal(size=(len(all_tm), 9))
    
    # apply 17 mag as the saturation limit
    for f in range(9):
        tx = np.where(all_tm[:,f] < 17.0000)[0]
        all_test_me[tx] = np.nan
        all_test_m[tx] = np.nan
        all_train_me[tx] = np.nan
        all_train_m[tx] = np.nan
        del tx

    # do not allow an "upscattering" of > 0.2 mag
    for f in range(9):
        tx = np.where(all_tm[:,f] > np_test_m5[f] + 0.20)[0]
        all_test_me[tx] = np.nan
        all_test_m[tx] = np.nan
        del tx
        tx = np.where(all_tm[:,f] > np_train_m5[f] + 0.20)[0]
        all_train_me[tx] = np.nan
        all_train_m[tx] = np.nan
        del tx

    # apply magnitude cuts
    for f in range(9):
        te_x = np.where(all_test_m[:,f] > np_test_mcut[f])[0]
        if len(te_x) > 0:
            all_test_m[te_x, f] = np.nan
            all_test_me[te_x, f] = np.nan
            if (force_idet == True) & (f == 3):
                all_test_m[te_x, :] = np.nan
                all_test_me[te_x, :] = np.nan
            if (force_gridet == True) & ((f == 1) | (f == 2) | (f == 3)):
                all_test_m[te_x, :] = np.nan
                all_test_me[te_x, :] = np.nan
        tr_x = np.where(all_train_m[:,f] > np_train_mcut[f])[0]
        if len(tr_x) > 0:
            all_train_m[tr_x, f] = np.nan
            all_train_me[tr_x, f] = np.nan
            if (force_idet == True) & (f == 3):
                all_train_m[tr_x, :] = np.nan
                all_train_me[tr_x, :] = np.nan
            if (force_gridet == True) & ((f == 1) | (f == 2) | (f == 3)):
                all_train_m[tr_x, :] = np.nan
                all_train_me[tr_x, :] = np.nan
        del te_x,tr_x
        
    # apply filtmask
    for f, fm in enumerate(filtmask):
        if fm == 0:
            all_test_m[:, f] = np.nan
            all_test_me[:, f] = np.nan
            all_train_m[:, f] = np.nan
            all_train_me[:, f] = np.nan

    # calculate colors, color errors, and number of colors
    all_test_c = np.zeros((len(all_tm), 8), dtype='float')
    all_test_ce = np.zeros((len(all_tm), 8), dtype='float')
    all_train_c = np.zeros((len(all_tm), 8), dtype='float')
    all_train_ce = np.zeros((len(all_tm), 8), dtype='float')
    for c in range(8):
        all_test_c[:, c]   = all_test_m[:, c] - all_test_m[:, c+1]
        all_train_c[:, c]  = all_train_m[:, c] - all_train_m[:, c+1]
        all_test_ce[:, c]  = np.sqrt( all_test_me[:, c]**2  + all_test_me[:, c+1]**2 )
        all_train_ce[:, c] = np.sqrt( all_train_me[:, c]**2 + all_train_me[:, c+1]**2 )
    all_test_Nc = np.nansum(all_test_c/all_test_c, axis=1)
    all_train_Nc = np.nansum(all_train_c/all_train_c, axis=1)

    # create test and training sets
    te_x = np.where( all_test_Nc >= cmnn_minNc )[0]
    tr_x = np.where( all_train_Nc >= cmnn_minNc )[0]

    if (len(te_x) < test_N) | (len(tr_x) < train_N):
        print('Error. Desired number of test/training galaxies higher than what is available.')
        print('  test number desired, available: %i %i' % (test_N, len(te_x)))
        print('  train number desired, available: %i %i' % (train_N, len(tr_x)))
        print('Exit (inputs too constraining to build test/train set).')
        exit()
    
    else:
        te_rx = np.random.choice(te_x, size=test_N, replace=False)
        test_fout = open('output/run_'+runid+'/test.cat', 'w')
        for i in te_rx:
            test_fout.write('%10i %10.8f ' % (all_id[i], all_tz[i]))
            for f in range(9):
                test_fout.write('%9.6f %9.6f ' % (all_test_m[i, f], all_test_me[i, f]))
            for c in range(8):
                test_fout.write('%9.6f %9.6f ' % (all_test_c[i, c], all_test_ce[i, c]))
            test_fout.write('\n')
        test_fout.close()
        del te_rx,test_fout

        tr_rx = np.random.choice(tr_x, size=train_N, replace=False)
        train_fout = open('output/run_'+runid+'/train.cat','w')
        for i in tr_rx:
            train_fout.write('%10i %10.8f ' % (all_id[i], all_tz[i]))
            for f in range(9):
                train_fout.write('%9.6f %9.6f ' % (all_train_m[i, f], all_train_me[i, f]))
            for c in range(8):
                train_fout.write('%9.6f %9.6f ' % (all_train_c[i, c], all_train_ce[i, c]))
            train_fout.write('\n')
        train_fout.close()
        del tr_rx,train_fout

        if verbose:
            print('Wrote ','output/run_'+runid+'/test.cat, output/run_'+runid+'/train.cat')
            print('Finished cmnn_catalog.make_test_and_train: ',datetime.datetime.now())


def make_plots(verbose, runid, filtmask):

    if verbose:
        print('Starting cmnn_catalog.make_plots: ',datetime.datetime.now())

    if os.path.isdir('output/run_'+runid+'/plot_cats') == False:
        os.system('mkdir output/run_'+runid+'/plot_cats')

    fnm = 'output/run_'+runid+'/test.cat'
    test_tz = np.loadtxt( fnm, dtype='float', usecols=(1))
    test_m  = np.loadtxt( fnm, dtype='float', usecols=(2,4,6,8,10,12,14,16,18))
    test_me = np.loadtxt( fnm, dtype='float', usecols=(3,5,7,9,11,13,15,17,19))

    fnm = 'output/run_'+runid+'/train.cat'
    train_tz = np.loadtxt( fnm, dtype='float', usecols=(1))
    train_m  = np.loadtxt( fnm, dtype='float', usecols=(2,4,6,8,10,12,14,16,18))
    train_me = np.loadtxt( fnm, dtype='float', usecols=(3,5,7,9,11,13,15,17,19))

    # redshift
    pfnm = 'output/run_'+runid+'/plot_cats/hist_ztrue'
    fig = plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size':20})
    plt.hist( test_tz,  density=True, bins=30, histtype='step', ls='solid', lw=1, \
             alpha=1, color='black', label='test')
    plt.hist( train_tz, density=True, bins=30, histtype='step', ls='solid', lw=4, \
             alpha=0.4, color='black', label='train')
    plt.xlabel('Mock Catalog True Redshift')
    plt.ylabel('Fraction of Galaxies')
    plt.legend(loc='upper right', prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm, bbox_inches='tight')
    if verbose: print('Wrote '+pfnm)

    # magnitude
    filtnames = ['u','g','r','i','z','y','J','H','K']
    for f, fm in enumerate(filtmask):
        if fm == 1:
            pfnm = 'output/run_'+runid+'/plot_cats/hist_mag_'+filtnames[f]
            fig  = plt.figure(figsize=(10,7))
            plt.rcParams.update({'font.size':20})
            tex = np.where(np.isfinite(test_m[:, f]))[0]
            plt.hist(test_m[tex, f], density=True, cumulative=True, bins=30, histtype='step', \
                     ls='solid', lw=1, alpha=1, color='black', label='test '+filtnames[f])
            trx = np.where(np.isfinite(train_m[:, f]))[0]
            plt.hist(train_m[trx, f], density=True, cumulative=True, bins=30, histtype='step', \
                     ls='solid', lw=4, alpha=0.4, color='black', label='train '+filtnames[f])
            del tex, trx
            plt.xlabel('Observed Apparent '+filtnames[f]+'-band Magnitude')
            plt.ylabel('Cumulative Fraction of Galaxies')
            plt.legend(loc='upper left', prop={'size':16}, labelspacing=0.5)
            plt.savefig(pfnm, bbox_inches='tight')
            if verbose: print('Wrote '+pfnm)
    
    # error vs magnitude
    for f, fm in enumerate(filtmask):
        if fm == 1:
            pfnm = 'output/run_'+runid+'/plot_cats/mage_vs_mag_'+filtnames[f]
            fig  = plt.figure(figsize=(10,7))
            plt.rcParams.update({'font.size':20})
            tex = np.where(np.isfinite(test_m[:, f]))[0]
            tx = np.random.choice(tex, size=5000, replace=False)
            plt.plot(test_m[tx, f], test_me[tx, f], 'o', ms=3, alpha=0.4, mew=0, \
                     color='black', label='test '+filtnames[f])
            del tex, tx
            trx = np.where(np.isfinite(train_m[:, f]))[0]
            tx = np.random.choice(trx, size=5000, replace=False)
            plt.plot(train_m[tx, f], train_me[tx, f], 'o', ms=1, alpha=1, mew=0, \
                     color='black', label='train '+filtnames[f])
            del trx, tx
            plt.xlabel('Observed Apparent '+filtnames[f]+'-band Magnitude')
            plt.ylabel('Error')
            plt.title('5000 Random Galaxies')
            plt.legend(loc='upper left', prop={'size':16}, labelspacing=0.5)
            plt.savefig(pfnm, bbox_inches='tight')
            if verbose: print('Wrote '+pfnm)

    if verbose:
        print('Finshed cmnn_catalog.make_plots: ',datetime.datetime.now())

import os
import argparse
import datetime
import cmnn_catalog
import cmnn_photoz
import cmnn_analysis
# import cmnn_tools


def run(verbose, runid, test_m5, train_m5, test_mcut, train_mcut, force_idet, test_N, train_N, \
    cmnn_minNc, cmnn_minNN, cmnn_ppf, cmnn_rsel, cmnn_ppmag, cmnn_ppclr, stats_COR):

    ### Run each of the three components of the CMNN Estimator in turn:
    ###  1. Create test and training set catalogs based on user input.
    ###  2. Estimate photo-z with user-supplied parameters.
    ###  3. Calculate statistical measures and make analysis plots.

    ### 1. Create the catalogs, and make simple histograms
    os.system("echo 'Start cmnn_catalog.make_test_and_train(): "+str(datetime.datetime.now())+\
        "' >> output/run_"+args.user_runid+"/timestamps.dat")
    cmnn_catalog.make_test_and_train(verbose, runid, test_m5, train_m5, test_mcut, train_mcut, \
        force_idet, test_N, train_N, cmnn_minNc)

    os.system("echo 'Start cmnn_catalog.make_plots(): "+str(datetime.datetime.now())+\
        "' >> output/run_"+args.user_runid+"/timestamps.dat")
    cmnn_catalog.make_plots(verbose, runid)

    ### 2. Estimate photometric redshifts
    os.system("echo 'Start cmnn_photoz.make_zphot(): "+str(datetime.datetime.now())+\
        "' >> output/run_"+args.user_runid+"/timestamps.dat")
    cmnn_photoz.make_zphot(verbose, runid, force_idet, cmnn_minNc, cmnn_minNN, cmnn_ppf, cmnn_rsel, \
        cmnn_ppmag, cmnn_ppclr)

    ### 3. Analyse the photo-z estimates (make statistics and plots of the results)
    os.system("echo 'Start cmnn_analysis.make_stats_file(): "+str(datetime.datetime.now())+\
        "' >> output/run_"+args.user_runid+"/timestamps.dat")
    cmnn_analysis.make_tzpz_plot(verbose, runid)
    cmnn_analysis.make_stats_file(verbose, runid, stats_COR)
    cmnn_analysis.make_stats_plots(verbose=verbose, runid=runid)


if __name__ == '__main__':

    ### Parse the input for the user-supplied magnitude limits and redshift
    ### Pass inputs to cmnn_run.run()

    parser = argparse.ArgumentParser(description='Allow for passing of arguments.')

    ### Define all allowed user input

    ### Argument:    verbose, type bool, default value False
    ### Description: if True, prints more intermediate information to the screen
    ### Example:     python cmnn_run.py --verbose False
    parser.add_argument('--verbose', action='store', dest='user_verbose', type=bool, \
        help='print more information to the screen', default=True)

    ### Argument:    runid, type str, default value 1
    ### Description: unique run identifier for labeling the output files
    ### Example:     python cmnn_run.py --runid 2
    ### Example:     python cmnn_run.py --runid helloworld
    parser.add_argument('--runid', action='store', dest='user_runid', type=str, \
        help='run identifier for output', default='1')

    ### Argument:    clobber, type bool 1, default False
    ### Description: if True, overwrites any existing output for this runid
    ### Example:     python cmnn_run.py --clobber True
    parser.add_argument('--clobber', action='store', dest='user_clobber', type=bool, \
        help='overwrite existing output for given runid', default=False, choices=[True,False])

    ### Argument:    test_m5, type float 6, default 26.1 27.4 27.5 26.8 26.1 24.9 (baseline 10-year depth)
    ### Description: the 5-sigma magnitude limits (depths) to apply to the test-set galaxies
    ### Example:     python cmnn_run.py --test_m5 23.9 25.0 24.7 24.0 23.3 22.1
    parser.add_argument('--test_m5', nargs='+', action='store', dest='user_test_m5', type=float,\
        help='test-set 5sig mag depths u g r i z y', default=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9])
    
    ### Argument:    train_m5, type float 6, default 26.1 27.4 27.5 26.8 26.1 24.9 (baseline 10-year depth)
    ### Description: the 5-sigma magnitude limits (depths) to apply to the train-set galaxies
    ### Example:     python cmnn_run.py --train_m5 27.0 28.0 28.0 27.0 27.0 25.0
    parser.add_argument('--train_m5', nargs='+', action='store', dest='user_train_m5', type=float,\
        help='train-set 5sig mag depths u g r i z y', default=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9])

    ### Argument:    test_mcut, type float 6, default 26.1 27.4 27.5 25.0 26.1 24.9 (baseline 10-year depth, i<25)
    ### Description: a magnitude cut-off to apply to the test-set galaxies
    ### Example:     python cmnn_run.py --test_mcut 23.9 25.0 24.7 24.0 23.3 22.1
    parser.add_argument('--test_mcut', nargs='+', action='store', dest='user_test_mcut', type=float,\
        help='test-set mag det cuts u g r i z y', default=[26.1, 27.4, 27.5, 25.0, 26.1, 24.9])
    
    ### Argument:    train_mcut, type float 6, default 26.1 27.4 27.5 25.0 26.1 24.9 (baseline 10-year depth, i<25)
    ### Description: a magnitude cut-off to apply to the train-set galaxies
    ### Example:     python cmnn_run.py --train_mcut 27.0 28.0 28.0 25.0 27.0 25.0
    parser.add_argument('--train_mcut', nargs='+', action='store', dest='user_train_mcut', type=float,\
        help='train-set mag det cuts u g r i z y', default=[26.1, 27.4, 27.5, 25.0, 26.1, 24.9])

    ### Argument:    force_idet, type bool 1, default True
    ### Description: force detection in i-band for all test and train galaxies
    ### Example:     python cmnn_run.py --force_idet False
    parser.add_argument('--force_idet', action='store', dest='user_force_idet', type=bool, \
        help='force i-band detection for all galaxies', default=True, choices=[True,False])

    ### Argument:    test_N, type int 1, default 40000
    ### Description: number of test-set galaxies
    ### Example:     python cmnn_run.py --test_N 50000
    parser.add_argument('--test_N', action='store', dest='user_test_N', type=int, \
        help='number of test-set galaxies', default=40000)

    ### Argument:    train_N, type int 1, default 200000
    ### Description: number of train-set galaxies
    ### Example:     python cmnn_run.py --train_N 500000
    parser.add_argument('--train_N', action='store', dest='user_train_N', type=int, \
        help='number of train-set galaxies', default=200000)

    ### Argument:    cmnn_minNc, type int 1, default 3, accepted values 2 to 5
    ### Description: minimum number of colors required for inclusion in catalog
    ### Example:     python cmnn_run.py --cmnn_minNc 5
    parser.add_argument('--cmnn_minNc', action='store', dest='user_cmnn_minNc', type=int, \
        help='CMNN: minimum number of colors for galaxies', default=3, choices=[2,3,4,5])

    ### Argument:    cmnn_minNN, type int 1, default 10, accepted values 0 to 20
    ### Description: forced minimum number of training-set galaxies in the CMNN subset
    ### Example:     python cmnn_run.py --cmnn_minNN 5
    parser.add_argument('--cmnn_minNN', action='store', dest='user_cmnn_minNN', type=int, \
        help='CMNN: minimum number of nearest neighbors', default=10, choices=range(21))

    ### Argument:    cmnn_ppf, type float 1, default 0.68, accepted values 0.68 or 0.95
    ### Description: the percent point function that defines the Mahalanobis distance threshold of the CMNN
    ### Example:     python cmnn_run.py --cmnn_ppf 0.95
    parser.add_argument('--cmnn_ppf', action='store', dest='user_cmnn_ppf', type=float, \
        help='CMNN: percent point function value (0.68 or 0.95)', default=0.68, choices=[0.68,0.95])

    ### Argument:    cmnn_rsel, type int 1, default 2, accepted values 0, 1, or 2
    ### Description: mode of random selection of a training-set galaxy from the CMNN subset:
    ###               0 : random
    ###               1 : nearest neighbor (lowest Mahalanobis distance)
    ###               2 : random weighted by inverse of Mahalanobis distance
    ### Example:     python cmnn_run.py --cmnn_rsel 0
    parser.add_argument('--cmnn_rsel', action='store', dest='user_cmnn_rsel', type=int, \
        help='CMNN: mode of random selection from CMNN subset (0, 1, or 2)', default=2, choices=[0,1,2])

    ### Argument:    cmnn_ppmag, type bool 1, default False
    ### Description: apply a "pseudo-prior" to the training set based on the test-set's i-band magnitude
    ### Example:     python cmnn_run.py --cmnn_ppmag True
    parser.add_argument('--cmnn_ppmag', action='store', dest='user_cmnn_ppmag', type=bool, \
        help='CMNN: apply magnitude pre-cut to training set', default=False, choices=[True,False])

    ### Argument:    cmnn_ppclr, type bool 1, default True
    ### Description: apply a "pseudo-prior" to the training set based on the test-set's g-r and r-i color
    ### Example:     python cmnn_run.py --cmnn_ppclr False
    parser.add_argument('--cmnn_ppclr', action='store', dest='user_cmnn_ppclr', type=bool, \
        help='CMNN: apply color pre-cut to training set', default=True, choices=[True,False])

    ### Argument:    stats_COR, type float 1, default 1.5
    ### Description: reject galaxies with (ztrue-zphot)/(1+zphot) > VALUE from the stddev and bias
    ### Example:     python cmnn_run.py --stats_COR 2.0
    parser.add_argument('--stats_COR', action='store', dest='user_stats_COR', type=float, \
        help='Stats: define catastrophic outliers rejected (COR) from stddev and bias', default=1.5)

    args = parser.parse_args()

    ### Clobber output if desired (clear output directory)
    if os.path.exists('output/run_'+args.user_runid):
        if args.user_clobber == True:        
            if args.user_verbose: print('Clobbering output/run_'+args.user_runid)
            os.system('rm -rf output/run_'+args.user_runid)
        else:
            print('Error. Output file already exists : ','output/run_'+args.user_runid)
            print(' To overwrite, use --clobber_runid True')
            print('Exit (bad runid).')
            exit()

    os.system('mkdir output/run_'+args.user_runid)
    os.system('touch output/run_'+args.user_runid+'/timestamps.dat')   
    os.system("echo 'File initiated: "+str(datetime.datetime.now())+\
        "' >> output/run_"+args.user_runid+"/timestamps.dat")

    ### User must pass 6 magnitudes for each of the test/train depths/cuts
    if (len(args.user_test_m5) != 6) | (len(args.user_train_m5) != 6) | \
       (len(args.user_test_mcut) != 6) | (len(args.user_train_mcut) != 6):
        print('Error. Input magnitude lists must have six elements.')
        print('  test_m5    : %i ' % len(args.user_test_m5))
        print('  train_m5   : %i ' % len(args.user_train_m5))
        print('  test_mcut  : %i ' % len(args.user_test_mcut))
        print('  train_mcut : %i ' % len(args.user_train_mcut))
        print('Exit (bad mag inputs).')
        exit()

    ### Check the other inputs
    fail = False

    ### Set the minimum and maximum magnitudes allowed for simulating LSST data
    ###   m5_min   : minimum 5-sigma depths set to a single standard visit (30 second integration)
    ###   m5_max   : maximum 5-sigma depths set to 29th mag for all filters (edge of reason)
    ###   mcut_min : minimum cut set to near saturation for a single standard visit (30 sec)
    ###   mcut_max : =m5_max, except i<25 mag to match SRD "gold sample" (and match provided catalog)
    filters  = ['u','g','r','i','z','y']
    m5_min   = [23.9, 25.0, 24.7, 24.0, 23.3, 22.1]
    m5_max   = [29.0, 29.0, 29.0, 29.0, 29.0, 29.0]
    mcut_min = [17.0, 17.0, 17.0, 17.0, 17.0, 17.0]
    mcut_max = [29.0, 29.0, 29.0, 25.0, 29.0, 29.0]
    mfail = False
    message = ''
    for f in range(6):
        if (args.user_test_m5[f] < m5_min[f]) | (args.user_test_m5[f] > m5_max[f]):
            message += '  test_m5: filter, value, min, max = %s %4.2f %4.2f %4.2f \n' % \
            (filters[f],args.user_test_m5[f],m5_min[f],m5_max[f])
            mfail = True
        if (args.user_train_m5[f] < m5_min[f]) | (args.user_train_m5[f] > m5_max[f]):
            message += '  train_m5: filter, value, min, max = %s %4.2f %4.2f %4.2f \n' % \
            (filters[f],args.user_train_m5[f],m5_min[f],m5_max[f])
            mfail = True
        if (args.user_test_mcut[f] < mcut_min[f]) | (args.user_test_mcut[f] > mcut_max[f]):
            message += '  test_mcut: filter, value, min, max = %s %4.2f %4.2f %4.2f \n' % \
            (filters[f],args.user_test_mcut[f],mcut_min[f],mcut_max[f])
            mfail = True
        if (args.user_train_mcut[f] < mcut_min[f]) | (args.user_train_mcut[f] > mcut_max[f]):
            message += '  train_mcut: filter, value, min, max = %s %4.2f %4.2f %4.2f \n' % \
            (filters[f],args.user_train_mcut[f],mcut_min[f],mcut_max[f])
            mfail = True
    if mfail == True:
        print('Error. Input value(s) for limiting magnitude(s) are out of accepted range(s).')
        print(message)
        fail = True
    del filters, m5_min,m5_max,mcut_min,mcut_max, mfail,message

    if args.user_stats_COR <= 0:
        print('Error. Input value for stats_COR must be greater than zero.')
        print('  stats_COR : %4.2f \n' % args.user_stats_COR)
        fail = True

    if (args.user_test_N <= 0) | (args.user_test_N > 100000):
        print('Error. Input value for test_N must be between 1 and 100000.')
        print('  test_N : %i \n' % args.user_test_N)
        fail = True

    if (args.user_train_N < 50000) | (args.user_train_N > 1000000):
        print('Error. Input value for train_N must be between 50000 and 1000000.')
        print('  train_N : %i \n' % args.user_train_N)
        fail = True

    if (args.user_cmnn_ppmag == True) & (args.user_force_idet == False):
        print('Error. Must set force_idet=True in order to set cmnn_ppmag=True.')
        print('  cmnn_ppmag : %r \n' % args.user_cmnn_ppmag)
        print('  force_idet : %r \n' % args.user_force_idet)
        fail = True

    if fail:
        print('Exit (bad user inputs, as listed above).')
        exit()
    del fail

    if args.user_verbose: print('Done checking user input, all have passed.')

    ### Print values of arguments to screen for user.
    if args.user_verbose:
        print(' ')
        print('User inputs (can also be found in output/run_'+args.user_runid+'):')
        print( '%-11s %s' % ('runid',args.user_runid) )
        print( '%-11s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f' % \
            ('test_m5',args.user_test_m5[0],args.user_test_m5[1],args.user_test_m5[2],\
                args.user_test_m5[3],args.user_test_m5[4],args.user_test_m5[5]) )
        print( '%-11s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f' % \
            ('train_m5',args.user_train_m5[0],args.user_train_m5[1],args.user_train_m5[2],\
                args.user_train_m5[3],args.user_train_m5[4],args.user_train_m5[5]) )
        print( '%-11s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f' % \
            ('test_mcut',args.user_test_mcut[0],args.user_test_mcut[1],args.user_test_mcut[2],\
                args.user_test_mcut[3],args.user_test_mcut[4],args.user_test_mcut[5]) )
        print( '%-11s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f' % \
            ('train_mcut',args.user_train_mcut[0],args.user_train_mcut[1],args.user_train_mcut[2],\
                args.user_train_mcut[3],args.user_train_mcut[4],args.user_train_mcut[5]) )
        print( '%-11s %r' % ('force_idet',args.user_force_idet) )
        print( '%-11s %i' % ('test_N',args.user_test_N) )
        print( '%-11s %i' % ('train_N',args.user_train_N) )
        print( '%-11s %i' % ('cmnn_minNc',args.user_cmnn_minNc) )
        print( '%-11s %i' % ('cmnn_minNN',args.user_cmnn_minNN) )
        print( '%-11s %4.2f' % ('cmnn_ppf',args.user_cmnn_ppf) )
        print( '%-11s %i' % ('cmnn_rsel',args.user_cmnn_rsel) )
        print( '%-11s %r' % ('cmnn_ppmag',args.user_cmnn_ppmag) )
        print( '%-11s %r' % ('cmnn_ppclr',args.user_cmnn_ppclr) )
        print( '%-11s %4.2f' % ('stats_COR',args.user_stats_COR) )

    ### Record user inputs to file.
    fout = open('output/run_'+args.user_runid+'/inputs.txt','w')
    fout.write( '%-11s %s \n' % ('runid',args.user_runid) )
    fout.write( '%-11s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f \n' % \
        ('test_m5',args.user_test_m5[0],args.user_test_m5[1],args.user_test_m5[2],\
            args.user_test_m5[3],args.user_test_m5[4],args.user_test_m5[5]) )
    fout.write( '%-11s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f \n' % \
        ('train_m5',args.user_train_m5[0],args.user_train_m5[1],args.user_train_m5[2],\
            args.user_train_m5[3],args.user_train_m5[4],args.user_train_m5[5]) )
    fout.write( '%-11s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f \n' % \
        ('test_mcut',args.user_test_mcut[0],args.user_test_mcut[1],args.user_test_mcut[2],\
            args.user_test_mcut[3],args.user_test_mcut[4],args.user_test_mcut[5]) )
    fout.write( '%-11s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f \n' % \
        ('train_mcut',args.user_train_mcut[0],args.user_train_mcut[1],args.user_train_mcut[2],\
            args.user_train_mcut[3],args.user_train_mcut[4],args.user_train_mcut[5]) )
    fout.write( '%-11s %r \n' % ('force_idet',args.user_force_idet) )
    fout.write( '%-11s %i \n' % ('test_N',args.user_test_N) )
    fout.write( '%-11s %i \n' % ('train_N',args.user_train_N) )
    fout.write( '%-11s %i \n' % ('cmnn_minNc',args.user_cmnn_minNc) )
    fout.write( '%-11s %i \n' % ('cmnn_minNN',args.user_cmnn_minNN) )
    fout.write( '%-11s %4.2f \n' % ('cmnn_ppf',args.user_cmnn_ppf) )
    fout.write( '%-11s %i \n' % ('cmnn_rsel',args.user_cmnn_rsel) )
    fout.write( '%-11s %r \n' % ('cmnn_ppmag',args.user_cmnn_ppmag) )
    fout.write( '%-11s %r \n' % ('cmnn_ppclr',args.user_cmnn_ppclr) )
    fout.write( '%-11s %4.2f \n' % ('stats_COR',args.user_stats_COR) )
    fout.close()
    if args.user_verbose: print('Wrote user inputs to output/run_'+args.user_runid+'/inputs.txt')

    ### Pass user input to run()
    if args.user_verbose: print(' ')
    if args.user_verbose: print('Starting cmnn_run.run: ', datetime.datetime.now())
    os.system("echo 'Start cmnn_run.run():  "+str(datetime.datetime.now())+\
        "' >> output/run_"+args.user_runid+"/timestamps.dat")
    run( args.user_verbose, args.user_runid, \
        args.user_test_m5, args.user_train_m5, args.user_test_mcut, args.user_train_mcut, \
        args.user_force_idet, \
        args.user_test_N, args.user_train_N, \
        args.user_cmnn_minNc, args.user_cmnn_minNN, args.user_cmnn_ppf, args.user_cmnn_rsel, \
        args.user_cmnn_ppmag, args.user_cmnn_ppclr, \
        args.user_stats_COR )
    if args.user_verbose: print('Finished cmnn_run.run: ', datetime.datetime.now())
    os.system("echo 'Finished cmnn_run.run(): "+str(datetime.datetime.now())+\
        "' >> output/run_"+args.user_runid+"/timestamps.dat")


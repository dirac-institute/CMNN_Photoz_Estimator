import os
import argparse
import datetime
import cmnn_catalog
import cmnn_photoz
import cmnn_analysis
import cmnn_tools

def run(verbose, runid, filtmask, catalog, \
        test_m5, train_m5, test_mcut, train_mcut, \
        force_idet, force_gridet, test_N, train_N, \
        cmnn_minNc, cmnn_minNN, cmnn_ppf, cmnn_rsel, \
        cmnn_ppmag, cmnn_ppclr, stats_COR):
    
    tmp_fnm = "output/run_"+runid+"/timestamps.dat"
    
    tmp_str = str(datetime.datetime.now())
    os.system("echo 'start cmnn_run: "+tmp_str+"' >> "+tmp_fnm)

    tmp_str = str(datetime.datetime.now())
    os.system("echo 'start cmnn_catalog: "+tmp_str+"' >> "+tmp_fnm)

    cmnn_catalog.make_test_and_train(verbose, runid, filtmask, \
                                     test_m5, train_m5, test_mcut, train_mcut, \
                                     force_idet, force_gridet, test_N, train_N, \
                                     cmnn_minNc, catalog)
    cmnn_catalog.make_plots(verbose, runid)

    tmp_str = str(datetime.datetime.now())
    os.system("echo 'start cmnn_photoz: "+tmp_str+"' >> "+tmp_fnm)
    
    cmnn_photoz.make_zphot(verbose, runid, filtmask, \
                           force_idet, force_gridet, \
                           cmnn_minNc, cmnn_minNN, cmnn_ppf, cmnn_rsel, \
                           cmnn_ppmag, cmnn_ppclr)

    tmp_str = str(datetime.datetime.now())
    os.system("echo 'start cmnn_analysis: "+tmp_str+"' >> "+tmp_fnm)
    
    cmnn_analysis.make_stats_file(verbose, runid, stats_COR)
    cmnn_analysis.make_stats_plots(verbose=verbose, runid=runid)
    cmnn_analysis.make_tzpz_plot(verbose, runid)
    cmnn_analysis.make_hist_plots(verbose, runid)

    tmp_str = str(datetime.datetime.now())
    os.system("echo 'finished cmnn_run: "+tmp_str+"' >> "+tmp_fnm)
    
    del tmp_fnm, tmp_str


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Allow for passing of arguments.')

    parser.add_argument('--catalog', action='store', dest='user_catalog', type=str, \
                        help='user-specified galaxy catalog with full path', \
                        default='Euclid-SDSS.txt')
    
    parser.add_argument('--verbose', action='store', dest='user_verbose', type=str2bool, \
                        help='print optional information to the screen', default=True)
    
    parser.add_argument('--runid', action='store', dest='user_runid', type=str, \
                        help='user-specified run identifier', default='myrunid')

    parser.add_argument('--clobber', action='store', dest='user_clobber', type=str2bool, \
                        help='overwrite existing output for given runid', default=False, \
                        choices=[True,False])
    
    parser.add_argument('--filtmask', nargs='+', action='store', dest='user_filtmask', type=int,\
                        help='set mask for use of filters u g r i z y J H K', \
                        default=[1, 1, 1, 1, 1, 1, 1, 1, 1])

    parser.add_argument('--yfilt', action='store', dest='user_yfilt', type=int, \
                        help='y-band filter: 0=PanSTARRS or 1=Euclid', default=0, \
                        choices=[0,1])
    
    parser.add_argument('--test_m5', nargs='+', action='store', dest='user_test_m5', type=float,\
                        help='5-sigma magnitude limits (depths) for test-set galaxies', \
                        default=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 24.0, 24.0, 24.0])
    
    parser.add_argument('--train_m5', nargs='+', action='store', dest='user_train_m5', type=float,\
                        help='5-sigma magnitude limits (depths) for training-set galaxies', \
                        default=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 24.0, 24.0, 24.0])

    parser.add_argument('--test_mcut', nargs='+', action='store', dest='user_test_mcut', type=float,\
                        help='magnitude cut-off to apply to the test-set galaxies', 
                        default=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 24.0, 24.0, 24.0])
    
    parser.add_argument('--train_mcut', nargs='+', action='store', dest='user_train_mcut', type=float,\
                        help='magnitude cut-off to apply to the training-set galaxies', \
                        default=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 24.0, 24.0, 24.0])
   
    parser.add_argument('--force_idet', action='store', dest='user_force_idet', type=str2bool, \
                        help='force i-band detection for all galaxies', default=True, \
                        choices=[True,False])

    parser.add_argument('--force_gridet', action='store', dest='user_force_gridet', type=str2bool, \
                        help='force g+r+i-band detection for all galaxies', default=True, \
                        choices=[True,False])

    parser.add_argument('--test_N', action='store', dest='user_test_N', type=int, \
                        help='number of test-set galaxies', default=40000)

    parser.add_argument('--train_N', action='store', dest='user_train_N', type=int, \
                        help='number of training-set galaxies', default=200000)

    parser.add_argument('--cmnn_minNc', action='store', dest='user_cmnn_minNc', type=int, \
                        help='CMNN: minimum number of colors for galaxies (2 to 8)', default=3, \
                        choices=[2,3,4,5,6,7,8])

    parser.add_argument('--cmnn_minNN', action='store', dest='user_cmnn_minNN', type=int, \
                        help='CMNN: minimum number of nearest neighbors (up to 21)', default=10, \
                        choices=range(21))

    parser.add_argument('--cmnn_ppf', action='store', dest='user_cmnn_ppf', type=float, \
                        help='CMNN: percent point function value (0.68 or 0.95)', default=0.680, \
                        choices=[0.680,0.950])

    # 0 : random
    # 1 : nearest neighbor (lowest Mahalanobis distance)
    # 2 : random weighted by inverse of Mahalanobis distance
    parser.add_argument('--cmnn_rsel', action='store', dest='user_cmnn_rsel', type=int, \
                        help='CMNN: mode of random selection from CMNN subset (0, 1, or 2)', default=2, \
                        choices=[0,1,2])

    parser.add_argument('--cmnn_ppmag', action='store', dest='user_cmnn_ppmag', type=str2bool, \
                        help='CMNN: apply magnitude pre-cut to training set', default=False, \
                        choices=[True,False])

    parser.add_argument('--cmnn_ppclr', action='store', dest='user_cmnn_ppclr', type=str2bool, \
                        help='CMNN: apply color pre-cut to training set', default=True, \
                        choices=[True,False])

    parser.add_argument('--stats_COR', action='store', dest='user_stats_COR', type=float, \
                        help='reject galaxies with (ztrue-zphot)/(1+zphot) > VALUE from the stddev and bias', \
                        default=1.5)

    args = parser.parse_args()
    
    # ensure the input catalog exists
    if (os.path.isfile(args.user_catalog) == False):
        print('Error. Mock galaxy catalog file is missing or misnamed:')
        print(args.user_catalog)
        print('Exit (missing input file).')
        exit()
    
    # set up output directory and files
    tmp_path = 'output/run_'+args.user_runid
    if os.path.isdir('output') == False:
        os.system('mkdir output')
    else:
        if os.path.exists(tmp_path):
            if args.user_clobber == True:        
                if args.user_verbose: print('Clobbering '+tmp_path)
                os.system('rm -rf '+tmp_path)
            else:
                print('Error. Output file already exists : '+tmp_path)
                print('To overwrite, use --clobber True')
                print('Exit (bad runid).')
                exit()
    os.system('mkdir '+tmp_path)

    tmp_fnm = tmp_path+'/timestamps.dat'
    tmp_str = str(datetime.datetime.now())
    os.system('touch '+tmp_fnm)
    os.system("echo 'file created: "+tmp_str+"' >> "+tmp_fnm)
    del tmp_fnm, tmp_str
    
    # ensure the number of filters
    Nfilts = 9
    if (len(args.user_test_m5) != Nfilts) | (len(args.user_train_m5) != Nfilts) | \
       (len(args.user_test_mcut) != Nfilts) | (len(args.user_train_mcut) != Nfilts) | \
       (len(args.user_filtmask) != Nfilts):
        print('Error. Input lists must have '+str(Nfilts)+' elements (one per filter u g r i z y J H K).')
        print('  filtmask   : %i ' % len(args.user_filtmask))
        print('  test_m5    : %i ' % len(args.user_test_m5))
        print('  train_m5   : %i ' % len(args.user_train_m5))
        print('  test_mcut  : %i ' % len(args.user_test_mcut))
        print('  train_mcut : %i ' % len(args.user_train_mcut))
        print('Exit (bad inputs).')
        exit()

    # check the other inputs
    fail = False

    filters  = ['u','g','r','i','z','y','J','H','K']
    m5_min   = [23.9, 25.0, 24.7, 24.0, 23.3, 22.1, 20.0, 20.0, 20.0]
    m5_max   = [29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0]
    mcut_min = [17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 16.0, 16.0, 16.0]
    mcut_max = [29.0, 29.0, 29.0, 25.0, 29.0, 29.0, 29.0, 29.0, 29.0]
    mfail = False
    message = ''
    for f in range(Nfilts):
        if (args.user_test_m5[f] < m5_min[f]) | (args.user_test_m5[f] > m5_max[f]):
            message += '  test_m5: filter, value, min, max = %s %6.3f %6.3f %6.3f \n' % \
            (filters[f], args.user_test_m5[f], m5_min[f], m5_max[f])
            mfail = True
        if (args.user_train_m5[f] < m5_min[f]) | (args.user_train_m5[f] > m5_max[f]):
            message += '  train_m5: filter, value, min, max = %s %6.3f %6.3f %6.3f \n' % \
            (filters[f], args.user_train_m5[f], m5_min[f], m5_max[f])
            mfail = True
        if (args.user_test_mcut[f] < mcut_min[f]) | (args.user_test_mcut[f] > mcut_max[f]):
            message += '  test_mcut: filter, value, min, max = %s %6.3f %6.3f %6.3f \n' % \
            (filters[f], args.user_test_mcut[f], mcut_min[f], mcut_max[f])
            mfail = True
        if (args.user_train_mcut[f] < mcut_min[f]) | (args.user_train_mcut[f] > mcut_max[f]):
            message += '  train_mcut: filter, value, min, max = %s %6.3f %6.3f %6.3f \n' % \
            (filters[f], args.user_train_mcut[f], mcut_min[f], mcut_max[f])
            mfail = True
    if mfail == True:
        print('Error. Input value(s) for limiting magnitude(s) are out of accepted range(s).')
        print(message)
        fail = True
    del filters, m5_min, m5_max, mcut_min, mcut_max, mfail, message

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

    if (args.user_cmnn_ppclr == True) & (args.user_force_gridet == False):
        print('Error. Must set force_gridet=True in order to set cmnn_ppclr=True.')
        print('  cmnn_ppclr : %r \n' % args.user_cmnn_ppclr)
        print('  force_gridet : %r \n' % args.user_force_gridet)
        fail = True

    if fail:
        print('Exit (bad user inputs, as listed above).')
        exit()
    del fail

    if args.user_verbose:
        print('Done checking user input, all have passed.')

    fout = open(tmp_path+'/inputs.txt', 'w')
    fout.write('%-11s %s \n' % ('catalog', args.user_catalog))
    fout.write('%-11s %r \n' % ('clobber', args.user_clobber))
    fout.write('%-11s %s \n' % ('runid', args.user_runid))
    fout.write('%-11s %1i %1i %1i %1i %1i %1i %1i %1i %1i \n' % \
               ('filtmask',\
                args.user_filtmask[0], args.user_filtmask[1], args.user_filtmask[2],\
                args.user_filtmask[3], args.user_filtmask[4], args.user_filtmask[5],\
                args.user_filtmask[6], args.user_filtmask[7], args.user_filtmask[8]))
    fout.write('%-11s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f \n' % \
               ('test_m5',\
                args.user_test_m5[0], args.user_test_m5[1], args.user_test_m5[2],\
                args.user_test_m5[3], args.user_test_m5[4], args.user_test_m5[5],\
                args.user_test_m5[6], args.user_test_m5[7], args.user_test_m5[8]))
    fout.write('%-11s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f \n' % \
               ('test_m5',\
                args.user_test_m5[0], args.user_test_m5[1], args.user_test_m5[2],\
                args.user_test_m5[3], args.user_test_m5[4], args.user_test_m5[5],\
                args.user_test_m5[6], args.user_test_m5[7], args.user_test_m5[8]))
    fout.write( '%-11s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f \n' % \
               ('train_m5',\
                args.user_train_m5[0], args.user_train_m5[1], args.user_train_m5[2],\
                args.user_train_m5[3], args.user_train_m5[4], args.user_train_m5[5],\
                args.user_train_m5[6], args.user_train_m5[7], args.user_train_m5[8]))
    fout.write('%-11s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f \n' % \
               ('test_mcut',\
                args.user_test_mcut[0], args.user_test_mcut[1], args.user_test_mcut[2],\
                args.user_test_mcut[3], args.user_test_mcut[4], args.user_test_mcut[5],\
                args.user_test_mcut[6], args.user_test_mcut[7], args.user_test_mcut[8]))
    fout.write('%-11s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f \n' % \
               ('train_mcut',\
                args.user_train_mcut[0], args.user_train_mcut[1], args.user_train_mcut[2],\
                args.user_train_mcut[3], args.user_train_mcut[4], args.user_train_mcut[5],\
                args.user_train_mcut[6], args.user_train_mcut[7], args.user_train_mcut[8]))
    fout.write('%-11s %r    \n' % ('force_idet', args.user_force_idet))
    fout.write('%-11s %r    \n' % ('force_gridet', args.user_force_gridet))
    fout.write('%-11s %i    \n' % ('test_N', args.user_test_N))
    fout.write('%-11s %i    \n' % ('train_N', args.user_train_N))
    fout.write('%-11s %i    \n' % ('cmnn_minNc', args.user_cmnn_minNc))
    fout.write('%-11s %i    \n' % ('cmnn_minNN', args.user_cmnn_minNN))
    fout.write('%-11s %5.3f \n' % ('cmnn_ppf', args.user_cmnn_ppf))
    fout.write('%-11s %i    \n' % ('cmnn_rsel', args.user_cmnn_rsel))
    fout.write('%-11s %r    \n' % ('cmnn_ppmag', args.user_cmnn_ppmag))
    fout.write('%-11s %r    \n' % ('cmnn_ppclr', args.user_cmnn_ppclr))
    fout.write('%-11s %4.2f \n' % ('stats_COR', args.user_stats_COR))
    fout.close()

    if args.user_verbose:
        print('Find user inputs in '+tmp_path+'/inputs.txt')
        print('Find user processing timestamps in '+tmp_path+'/timestamps.txt')
        print('Starting cmnn_run.run: ', datetime.datetime.now())
    del tmp_path
        
    run(args.user_verbose, args.user_runid, args.user_filtmask, args.user_catalog, \
        args.user_test_m5, args.user_train_m5, args.user_test_mcut, args.user_train_mcut, \
        args.user_force_idet, args.user_force_gridet, args.user_test_N, args.user_train_N, \
        args.user_cmnn_minNc, args.user_cmnn_minNN, args.user_cmnn_ppf, args.user_cmnn_rsel, \
        args.user_cmnn_ppmag, args.user_cmnn_ppclr, args.user_stats_COR)

    if args.user_verbose:
        print('Finished cmnn_run.run: ', datetime.datetime.now())
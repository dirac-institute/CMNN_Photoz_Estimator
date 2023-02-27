import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
import datetime


def get_stats(in_zspec, in_zphot, zlow, zhi, thresh_COR):
    
    '''
    Return the statistical measures for a given redshift bin bounded by zlow,zi.

    Inputs 
      in_zspec   : list of spectroscopic or 'true' redshifts for ALL test galaxies (0.3 to 3.0)
      in_zphot   : list of photometric redshift point estimates for ALL test galaxies (0.3 to 3.0)
      zlow       : low-end redshift of bin
      zhi        : high-end redshift of bin
      thresh_COR : zspec-zphot threshold to define "catastrophic outlier rejection"

    Definitions used in this function.
      dzo1pzp              : dz over 1 plus zphot, (zspec - zphot) / (1 + zphot)
      catastrophic outlier : test galaxy with | zspec - zphot | > stats_COR
      outlier              : test galaxy with (|dzo1pzp| > 3*IQRs) & (|dzo1pzp| > 0.06)
    The definition of an "outlier" comes from the SRD (which calls them 'catastrophic outliers'; ls.st/srd).
    For outliers, the IQRs used to identify them is calculated from all test galaxies in 0.3 < zphot < 3.0

    Definitions of the statistical measures.
      meanz   : the mean zphot of galaxies in bin
      fout    : fraction of outliers (see note below)
      stdd    : standard deviation in dzo1pzp of all galaxies in bin
      bias    : mean dzo1pzp of all galaxies in bin
      outbias : mean dzo1pzp of all outliers in bin
      IQR     : interquartile range of dzo1pzp
      IQRstdd : stdandard deviation from the IQR (= IQR / 1.349)
      IQRbias : bias of test galaxies in the IQR 
      COR     : a catastrophic outlier-rejected statistical measure

    Outputs returned
      meanz       : meanz
      CORmeanz    : post-COR, the meanz
      fout        : fout
      CORfout     : post-COR fout
      stdd        : stdd
      bias        : bias
      outbias     : outlier bias
      IQR         : IQR
      IQRstdd     : IQR stdd (robust standard deviatin)
      IQRbias     : IQR bias (robust bias)
      CORstdd     : post-COR stdd
      CORbias     : post-COR bias
      CORoutbias  : post-COR outlier bias
      CORIQR      : post-COR IQR
      CORIQRstdd  : post-COR IQR stdd
      CORIQRbias  : post-COR IQR bias
      estdd       : error in stdd
      ebias       : error in bias
      eoutbias    : error in outlier bias
      eIQR        : error in IQR
      eIQRstdd    : error in IQR stdd
      eIQRbias    : error in IQR bias
      eCORstdd    : error in post-COR stdd
      eCORbias    : error in post-COR bias
      eCORoutbias : error in post-COR outlier bias
      eCORIQR     : error in post-COR IQR
      eCORIQRstdd : error in post-COR IQR stdd
      eCORIQRbias : error in post-COR IQR bias
    '''

    # Bootstrap resample with replacement to estimate errors for the statistical measures.
    # Nmc : number of times to repeat measurement
    Nmc = int(1000)

    # Calculate the IQRs for all test galaxies with a zphot in the range 0.3 < zphot < 3.0.
    # Will need this to identify outliers using the SRD's definition of an outlier.
    #  fr = FULL RANGE (as in, all galaxies, not just this bin)
    tx = np.where((in_zphot >= float(0.300)) & (in_zphot <= float(3.000)))[0]
    fr_zspec = in_zspec[tx]
    fr_zphot = in_zphot[tx]
    fr_dzo1pzp = (fr_zspec - fr_zphot) / (float(1.0) + fr_zphot)
    q75, q25 = np.percentile(fr_dzo1pzp, [75 ,25])
    fr_IQRs = (q75 - q25) / float(1.349)
    del tx, fr_zphot, fr_zspec, fr_dzo1pzp, q75, q25

    # Same as above, but for catastrophic outlier rejected set of galaxies
    tx = np.where((in_zphot >= float(0.300)) & (in_zphot <= float(3.000)) & \
                  (np.abs(in_zphot - in_zspec) < thresh_COR))[0]
    fr_CORzspec = in_zspec[tx]
    fr_CORzphot = in_zphot[tx]
    fr_CORdzo1pzp = (fr_CORzspec - fr_CORzphot) / (float(1.0) + fr_CORzphot)
    q75, q25 = np.percentile(fr_CORdzo1pzp, [75 ,25])
    fr_CORIQRs = (q75 - q25) / float(1.349)
    del tx, fr_CORzphot, fr_CORzspec, fr_CORdzo1pzp, q75, q25

    # Identify all test galaxies in the requested bin.
    tx = np.where((in_zphot > zlow) & (in_zphot <= zhi))[0]
    zspec = in_zspec[tx]
    zphot = in_zphot[tx]
    del tx

    # Identify the subset of catastrophic outlier-rejected test galaxies in the requested bin
    tx = np.where((in_zphot > zlow) & (in_zphot <= zhi) & \
                  (np.abs(in_zphot - in_zspec) < thresh_COR))[0]
    CORzspec = in_zspec[tx]
    CORzphot = in_zphot[tx]
    del tx

    del in_zspec, in_zphot

    # Calculate the mean zphot in the bin
    meanz = np.mean(zphot)
    CORmeanz = np.mean(CORzphot)

    # Define bin_dzo1pzp for use in all stats
    dzo1pzp = (zspec - zphot) / (float(1.0) + zphot)
    CORdzo1pzp = (CORzspec - CORzphot) / (float(1.0) + CORzphot)

    # Fraction of outliers as defined by the SRD
    ox = np.where((np.fabs(dzo1pzp) > float(0.0600)) & \
                  (np.fabs(dzo1pzp) > (float(3.0)*fr_IQRs)))[0]
    fout = float(len(ox)) / float(len(dzo1pzp))
    del ox

    # Fraction of post-COR outliers; use same definition as fout and the SRD
    ox = np.where((np.fabs(CORdzo1pzp) > float(0.0600)) & \
                  (np.fabs(CORdzo1pzp) > (float(3.0)*fr_IQRs)))[0]
    CORfout = float(len(ox)) / float(len(CORdzo1pzp))
    del ox

    # Standard deviation
    stdd = np.std(dzo1pzp)
    CORstdd = np.std(CORdzo1pzp)

    # Bias
    bias = np.mean(dzo1pzp)
    CORbias = np.mean(CORdzo1pzp)

    # Outlier Bias
    ox = np.where((np.fabs(dzo1pzp) > float(0.0600)) & \
                  (np.fabs(dzo1pzp) > (float(3.0)*fr_IQRs)))[0]
    outbias = np.mean(dzo1pzp[ox])
    del ox
    ox = np.where((np.fabs(CORdzo1pzp) > float(0.0600)) & \
                  (np.fabs(CORdzo1pzp) > (float(3.0)*fr_CORIQRs)))[0]
    CORoutbias = np.mean(CORdzo1pzp[ox])
    del ox

    # Intraquartile Range
    q75, q25 = np.percentile(dzo1pzp, [75 ,25])
    IQR = (q75 - q25)
    IQRstdd = (q75 - q25) / float(1.349)
    tx = np.where((dzo1pzp > q25) & (dzo1pzp < q75))[0]
    IQRbias = np.mean(dzo1pzp[tx])
    del q75, q25, tx

    # COR Intraquartile Range
    q75, q25 = np.percentile(CORdzo1pzp, [75 ,25])
    CORIQR = (q75 - q25)
    CORIQRstdd = (q75 - q25) / float(1.349)
    tx = np.where((CORdzo1pzp > q25) & (CORdzo1pzp < q75))[0]
    CORIQRbias = np.mean(CORdzo1pzp[tx])
    del q75, q25, tx

    # Now do the MC and calculate errors for all quantities
    vals_s = np.zeros(Nmc, dtype='float')
    vals_b = np.zeros(Nmc, dtype='float')
    vals_ob = np.zeros(Nmc, dtype='float')
    vals_IQR = np.zeros(Nmc, dtype='float')
    vals_IQRs = np.zeros(Nmc, dtype='float')
    vals_IQRb = np.zeros(Nmc, dtype='float')
    vals_CORs = np.zeros(Nmc, dtype='float')
    vals_CORb = np.zeros(Nmc, dtype='float')
    vals_CORob = np.zeros(Nmc, dtype='float')
    vals_CORIQR = np.zeros(Nmc, dtype='float')
    vals_CORIQRs = np.zeros(Nmc, dtype='float')
    vals_CORIQRb = np.zeros(Nmc, dtype='float')
    for i in range(Nmc):
        tx = np.random.choice(len(dzo1pzp), size=len(dzo1pzp), replace=True, p=None)
        vals_s[i] = np.std(dzo1pzp[tx])
        vals_b[i] = np.mean(dzo1pzp[tx])
        q75, q25 = np.percentile(dzo1pzp[tx], [75 ,25])
        vals_IQR[i] = (q75 - q25)
        vals_IQRs[i] = (q75 - q25) / float(1.349)
        temp = dzo1pzp[tx]
        ttx = np.where((temp > q25) & (temp < q75))[0]
        vals_IQRb[i] = np.mean(temp[ttx])
        del q75, q25, temp, ttx
        tmp_dzo1pzp = dzo1pzp[tx]
        ox = np.where((np.fabs(tmp_dzo1pzp) > float(0.0600)) & \
            (np.fabs(tmp_dzo1pzp) > (float(3.0)*fr_IQRs)))[0]
        vals_ob[i] = np.mean(tmp_dzo1pzp[ox])
        del ox, tmp_dzo1pzp
        del tx

        tx = np.random.choice(len(CORdzo1pzp), size=len(CORdzo1pzp), replace=True, p=None)
        vals_CORs[i] = np.std(CORdzo1pzp[tx])
        vals_CORb[i] = np.mean(CORdzo1pzp[tx])
        q75, q25 = np.percentile(CORdzo1pzp[tx], [75 ,25])
        vals_CORIQR[i] = (q75 - q25)
        vals_CORIQRs[i] = (q75 - q25) / float(1.349)
        CORtemp = CORdzo1pzp[tx]
        ttx    = np.where((CORtemp > q25) & (CORtemp < q75))[0]
        vals_CORIQRb[i] = np.mean(CORtemp[ttx])
        del q75, q25, CORtemp, ttx
        tmp_CORdzo1pzp = CORdzo1pzp[tx]
        ox = np.where((np.fabs(tmp_CORdzo1pzp) > float(0.0600)) & \
            (np.fabs(tmp_CORdzo1pzp) > (float(3.0)*fr_CORIQRs)))[0]
        vals_CORob[i] = np.mean(tmp_CORdzo1pzp[ox])
        del ox, tmp_CORdzo1pzp
        del tx

    estdd = np.std(vals_s)
    ebias = np.std(vals_b)
    eoutbias = np.std(vals_ob)
    eIQR = np.std(vals_IQR)
    eIQRstdd = np.std(vals_IQRs)
    eIQRbias = np.std(vals_IQRb)
    eCORstdd = np.std(vals_CORs)
    eCORbias = np.std(vals_CORb)
    eCORoutbias = np.std(vals_CORob)
    eCORIQR = np.std(vals_CORIQR)
    eCORIQRstdd = np.std(vals_CORIQRs)
    eCORIQRbias = np.std(vals_CORIQRb)
    del vals_s, vals_b, vals_ob, vals_IQR, vals_IQRs, vals_IQRb
    del vals_CORs, vals_CORb, vals_CORob, vals_CORIQR, vals_CORIQRs, vals_CORIQRb

    return meanz, CORmeanz, fout, CORfout, stdd, bias, outbias, \
           IQR, IQRstdd, IQRbias, CORstdd, CORbias, CORoutbias, CORIQR, CORIQRstdd, CORIQRbias, \
           estdd, ebias, eoutbias, eIQR, eIQRstdd, eIQRbias, eCORstdd, eCORbias, eCORoutbias, \
           eCORIQR, eCORIQRstdd, eCORIQRbias


def make_stats_file(verbose, runid, stats_COR, input_zbins=[None], \
                    statsfile_suffix=None, bin_in_truez=False):

    '''
    Make a file containing the photo-z statistics in zphot bins.

    Input definitions 
      stats_COR    : zspec-zphot threshold to define "catastrophic outlier rejection"
      input_zbins  : an array of redshift bin centers [ [zlow_1,zhi_1], [zlow_2,zhi_2], ... [zlow_N,zhi_N] ]
      file_suffix  : an optional suffix to the output file to, e.g., denote user-supplied input bins 
      bin_in_truez : the bins are in true redshift (otherwise, default is to bin in photo-z)
    '''

    if verbose:
        print('Starting cmnn_analysis.make_stats_file: ', datetime.datetime.now())

    if os.path.isdir('output/run_'+runid+'/analysis') == False :
        os.system('mkdir output/run_'+runid+'/analysis')

    # Setup the redshift bins, the default is overlapping bins
    user_zbins = np.asarray(input_zbins, dtype='float')
    # If input_zbins == None, create the zbins
    if np.isnan(user_zbins).all():
        temp = []
        zlow = float(0.00)
        zhi = float(0.30)
        temp.append([zlow,zhi])
        for z in range(18):
            zlow += float(0.15)
            zhi  += float(0.15)
            temp.append([zlow,zhi])
        zbins = np.asarray(temp, dtype='float')
        del temp,zlow,zhi
    # If input_zbins != None, use the passed zbins
    else:
        zbins = user_zbins
    if verbose:
        print('zbins = ',zbins)

    # Read in the data
    if bin_in_truez == False:
        fnm = 'output/run_'+runid+'/zphot.cat'
        ztrue = np.loadtxt(fnm, dtype='float', usecols=(1))
        zphot = np.loadtxt(fnm, dtype='float', usecols=(2))

    # If you want to bin in TRUE z instead of PHOT z,
    #  just read the columns into the wrong array.
    if bin_in_truez == True:
        fnm = 'output/run_'+runid+'/zphot.cat'
        ztrue = np.loadtxt(fnm, dtype='float', usecols=(2))
        zphot = np.loadtxt(fnm, dtype='float', usecols=(1))

    # Open the output file for writing
    # Then loop over all zbins, obtain the statistical measures, and write them to file
    if bin_in_truez == False:
        ofnm = 'output/run_'+runid+'/analysis/stats.dat'
        if statsfile_suffix != None:
            ofnm = 'output/run_'+runid+'/analysis/stats_'+statsfile_suffix+'.dat'
    if bin_in_truez == True:
        ofnm = 'output/run_'+runid+'/analysis/stats_truezbins.dat'
        if statsfile_suffix != None:
            ofnm = 'output/run_'+runid+'/analysis/stats_'+statsfile_suffix+'_truezbins.dat'
    fout = open(ofnm,'w')
    fout.write('# zlow zhi '+\
        'meanz CORmeanz fout CORfout '+\
        'stdd bias outbias '+\
        'IQR IQRstdd IQRbias '+\
        'CORstdd CORbias CORoutbias '+\
        'CORIQR CORIQRstdd CORIQRbias '+\
        'estdd ebias eoutbias '+\
        'eIQR eIQRstdd eIQRbias '+\
        'eCORstdd eCORbias eCORoutbias '+\
        'eCORIQR eCORIQRstdd eCORIQRbias \n')
    for z in range(len(zbins)):
        stats = get_stats(ztrue, zphot, zbins[z,0], zbins[z,1], stats_COR)
        fout.write('%6.4f %6.4f ' % (zbins[z,0], zbins[z,1]))
        for s in range(28):
            fout.write('%6.4f ' % stats[s])
        fout.write('\n')
        del stats
    fout.close()
    if verbose: print('Wrote to: ',ofnm)


def make_stats_plots(verbose = True, runid = None, user_stats = ['fout', 'CORIQRstdd', 'CORIQRbias'], \
                     statsfile_suffix = None, bin_in_truez = False, \
                     show_SRD = True, show_binw = True, \
                     multi_run_ids = None, multi_run_labels = None, \
                     multi_run_colors = ['black', 'orange', 'green', 'darkviolet', 'blue', 'red'], \
                     plot_title = '', custom_pfnm_id = None, custom_axlims = None):

    '''
    Make plots for all the photo-z statistics for a given run.

    Input definitions 
      verbose          : extra output to screen
      runid            : the run to make plots for (ignored if multi_run_ids != None)
      user_stats       : array listing which stats to create plots for, default is ['fout','CORIQRstdd','CORIQRbias']
      statsfile_suffix : if not None, read file "stats_[suffix].dat" and add suffix to output plot names
      bin_in_truez     : if True, plot vs. true z (read file "stats_truezbins.dat" not "stats.dat")
      show_SRD         : True/False; if True, the SRD target values are shown as dashed horizontal lines
      show_binw        : True/False; if True, the bin widths are shown as light horizontal bars
      multi_run_ids    : array of multiple run ids to co-plot
      multi_run_labels : array of legend labels that describe each run
      multi_run_colors : array of color names to use for each run (six provided as defaults)
      plot_title       : add a title to the plot
      custom_pfmn_id   : custom plotname identifier (only for use instead of multiple runids)
      custom_axlims  : custom axes limits [[x1, x2], [y1, y2]]
    '''

    if verbose:
        print(' ')
        print('Starting cmnn_analysis.make_stats_plots(), ', datetime.datetime.now())

    if (multi_run_ids == None) & (runid == None):
        print(' ')
        print('Error in cmnn_analysis.make_stats_plots()')
        print('Need to specify input values for at least one of: runid or multi_run_ids.')
        print('  runid = ',runid)
        print('  multi_run_ids = ',multi_run_ids)
        print('Exit.')
        exit()
    if (multi_run_ids == None) & (runid != None):
        del multi_run_ids,multi_run_labels
        multi_run_ids = [runid]
        multi_run_labels = ['run '+runid]
        del runid
    if verbose:
        print('run ids: ', multi_run_ids)
        print('run labels: ', multi_run_labels)
    
    if (multi_run_ids == None) & (custom_pfnm_id != None):
        print(' ')
        print('Error, custom_pfnm_id is only for use with multi_run_ids.')
        print('  custom_pfnm_id: ', custom_pfnm_id)
        print('  multi_run_ids: ', multi_run_ids)
        print('Exit.')
        exit()

    # At this point, multi_run_ids is populated, let's make sure it's correct
    if (len(multi_run_ids) != len(multi_run_labels)):
        print(' ')
        print('Error in cmnn_analysis.make_stats_plots()')
        print('User-defined input regarding the multiple runs to plot is incompatible.')
        print('It is required that the lengths of the following arrays be equal:')
        print('  len(multi_run_ids) = ',    len(multi_run_ids))
        print('  len(multi_run_labels) = ', len(multi_run_labels))
        print('Exit.')
        exit()
    if (len(multi_run_ids) > len(multi_run_colors)):
        print(' ')
        print('Error in cmnn_analysis.make_stats_plots()')
        print('User-defined input regarding the multiple runs to plot is incompatible.')
        print('It is required that each run be assigned a unique color.')
        print('  len(multi_run_ids) = ',    len(multi_run_ids))
        print('  len(multi_run_colors) = ', len(multi_run_colors))
        print('The default list of colors is: ')
        print("  multi_run_colors=['black', 'orange', 'green', 'darkviolet', 'blue', 'red']")
        print('For example, if you have 7 runs to compare, you might want to pass:')
        print("  multi_run_colors=['black', 'orange', 'green', 'darkviolet', 'blue', 'red', 'brown']")
        print('Exit.')
        exit()
    for run_id in multi_run_ids:
        if bin_in_truez == False:
            stats_fnm = 'output/run_'+run_id+'/analysis/stats.dat'
            if statsfile_suffix != None:
                stats_fnm = 'output/run_'+run_id+'/analysis/stats_'+statsfile_suffix+'.dat'
        if bin_in_truez == True:
            stats_fnm = 'output/run_'+run_id+'/analysis/stats_truezbins.dat'
            if statsfile_suffix != None:
                stats_fnm = 'output/run_'+run_id+'/analysis/stats_'+statsfile_suffix+'_truezbins.dat'
        if os.path.exists(stats_fnm) == False:
            print(' ')
            print('Error, file does not exist: ',stats_fnm)
            print('Need to run cmnn_analysis.make_stats_file() first.')
            if bin_in_truez == True:
                print('   and set bin_in_truez == True ')
            print('Exit.')
            exit()
        del stats_fnm

    # Define needed parameters for ALL the possible statistical measures
    #  (but only the statistics specified in user_stats will have plots made)
    all_stats_names = np.asarray([\
        'fout','CORfout',\
        'stdd','bias','outbias',\
        'IQR','IQRstdd','IQRbias',\
        'CORstdd','CORbias','CORoutbias',\
        'CORIQR','CORIQRstdd','CORIQRbias'], dtype='str')
    all_stats_labels = np.asarray([\
        'Fraction of Outliers','(C.O.R.) Fraction of Outliers',\
        'Standard Deviation','Bias','Outlier Bias',\
        'IQR','Robust Standard Deviation','Robust Bias',\
        'Standard Deviation (C.O.R.)','Bias (C.O.R)','Outlier Bias (C.O.R.)',\
        'IQR (C.O.R.)','Robust Standard Deviation (C.O.R.)','Robust Bias (C.O.R.)'], dtype='str')
    # stats_xcols is whether to plot vs. meanz (col 2) or CORmeanz (col 3)
    all_stats_xcols = np.asarray([  2,  3,  2, 2, 2,  2, 2, 2,  3, 3, 3,  3, 3, 3], dtype='int')
    all_stats_ycols = np.asarray([  4,  5,  6, 7, 8,  9,10,11, 12,13,14, 15,16,17], dtype='int')
    all_stats_yecols = np.asarray([-99,-99, 17,18,19, 20,21,22, 23,24,15, 26,27,28], dtype='int')

    # to co-plot the SRD target values, need to know the type of statistic
    #   -1 : no corresponding SRD target value (i.e., for outbias)
    #    0 : a measure of the fraction of outliers
    #    1 : a standard deviation measure
    #    2 : a bias measure
    all_stats_srdtyp = np.asarray([  0,  0,  1, 2, -1,  1, 1, 2,  1, 2,-1,  1, 1, 2])

    # Match the input user_stats to the arrays of information fo plotting all the types of statistcs
    if user_stats == None:
        user_stats = ['fout','CORIQRstdd','CORIQRbias']
    x = []
    for stat in user_stats:
        tx = np.where(stat == all_stats_names)[0]
        if len(tx) == 1:
            x.append(tx[0])
        if verbose:
            if len(tx) == 0:
                print('WARNING: Unrecognized element in user_stats: ',stat,' no such plot will be made.')
        del tx
    sx = np.asarray(x, dtype='int')
    del x
    stats_names = all_stats_names[sx]
    stats_labels = all_stats_labels[sx]
    stats_xcols = all_stats_xcols[sx]
    stats_ycols = all_stats_ycols[sx]
    stats_yecols = all_stats_yecols[sx]
    stats_srdtyp = all_stats_srdtyp[sx]
    del sx,user_stats
    del all_stats_names,all_stats_labels,all_stats_xcols,all_stats_ycols,all_stats_yecols

    # Loop over all possible statistical measures and make a plot
    for s,stat in enumerate(stats_names):

        # Define which columns to be read from the stats file
        sxc = stats_xcols[s]
        syc = stats_ycols[s]

        # Initialize the plot
        fig = plt.figure(figsize=(12,8))
        plt.rcParams.update({'font.size':20})

        # Show the SRD target values, if desired
        if show_SRD == True:
            # Fraction of Outliers, 10%
            if stats_srdtyp[s] == 0:
                plt.plot([0.3,3.0], [0.10,0.10], lw=2, alpha=1, ls='dashed', color='black', label='SRD')
            # Standard Deviation, 0.02
            if stats_srdtyp[s] == 1:
                plt.plot([0.3,3.0], [0.02,0.02], lw=2, alpha=1, ls='dashed', color='black', label='SRD')
            # Bias, +/- 0.003
            if stats_srdtyp[s] == 2:
                plt.plot([0.3,3.0], [-0.003,-0.003], lw=2, alpha=1, ls='dashed', color='black', label='SRD')
                plt.plot([0.3,3.0], [0.003,0.003], lw=2, alpha=1, ls='dashed', color='black')

        pfnm_suffix = ''
        for r,run_id in enumerate(multi_run_ids):
            pfnm_suffix += '_'+run_id
            # Read in the statistical measures
            if bin_in_truez == False:
                stats_fnm = 'output/run_'+run_id+'/analysis/stats.dat'
                if statsfile_suffix != None:
                    stats_fnm = 'output/run_'+run_id+'/analysis/stats_'+statsfile_suffix+'.dat'
            if bin_in_truez == True:
                stats_fnm = 'output/run_'+run_id+'/analysis/stats_truezbins.dat'
                if statsfile_suffix != None:
                    stats_fnm = 'output/run_'+run_id+'/analysis/stats_'+statsfile_suffix+'_truezbins.dat'
            ## If we're showing the bins as horizontal bars
            if show_binw:
                binlo = np.loadtxt(stats_fnm, dtype='float', usecols=(0))
                binhi = np.loadtxt(stats_fnm, dtype='float', usecols=(1))
            xvals = np.loadtxt(stats_fnm, dtype='float', usecols=(sxc))
            yvals = np.loadtxt(stats_fnm, dtype='float', usecols=(syc))
            if stats_yecols[s] > 0:
                eyvals = np.loadtxt(stats_fnm, dtype='float', usecols=(stats_yecols[s]))
            # Plot the values, with error bars if appropriate
            plt.plot(xvals, yvals, lw=3, alpha=0.75, color=multi_run_colors[r], label=multi_run_labels[r])
            ## If we're showing the bins as horizontal bars
            if show_binw:
                for y,yv in enumerate(yvals):
                    plt.plot([binlo[y],binhi[y]], [yv,yv], lw=1, alpha=0.5, color=multi_run_colors[r])
                del binlo,binhi
            if stats_yecols[s] > 0:
                plt.errorbar(xvals, yvals, yerr=eyvals, fmt='none', elinewidth=3, \
                             capsize=3, capthick=3, ecolor=multi_run_colors[r])
                del eyvals
            del stats_fnm,xvals,yvals

        # Set axis parameters and write plot to file
        if bin_in_truez == False:
            plt.xlabel('Photometric Redshift')
        if bin_in_truez == True:
            plt.xlabel('True Redshift')
        plt.ylabel(stats_labels[s])
        legend=plt.legend(loc='best',numpoints=1,prop={'size':14},labelspacing=0.15) #,title=lgnd_title)
        # legend.get_title().set_fontsize('14')
        
        if custom_axlims != None:
            plt.xlim(custom_axlims[0])
            plt.ylim(custom_axlims[1])

        if len(multi_run_ids) == 1:
            if bin_in_truez == False:
                pfnm = 'output/run_'+multi_run_ids[0]+'/analysis/'+stat
                if statsfile_suffix != None:
                    pfnm = 'output/run_'+multi_run_ids[0]+'/analysis/'+stat+'_'+statsfile_suffix
            if bin_in_truez == True:
                pfnm = 'output/run_'+multi_run_ids[0]+'/analysis/'+stat+'_truezbins'
                if statsfile_suffix != None:
                    pfnm = 'output/run_'+multi_run_ids[0]+'/analysis/'+stat+'_'+statsfile_suffix+'_truezbins'
        if len(multi_run_ids) > 1:
            if os.path.exists('output/stats_plots') == False:
                os.system('mkdir output/stats_plots')        
            if bin_in_truez == False:
                pfnm = 'output/stats_plots/'+stat+pfnm_suffix
                if statsfile_suffix != None:
                    pfnm = 'output/stats_plots/'+stat+'_'+statsfile_suffix+pfnm_suffix
            if bin_in_truez == True:
                pfnm = 'output/stats_plots/'+stat+'_truezbins'+pfnm_suffix
                if statsfile_suffix != None:
                    pfnm = 'output/stats_plots/'+stat+'_'+statsfile_suffix+'_truezbins'+pfnm_suffix
            if custom_pfnm_id != None:
                pfnm = 'output/stats_plots/' + stat + '_' + custom_pfnm_id
            
        if plot_title != '':
            plt.title(plot_title)                                 

        plt.savefig(pfnm, bbox_inches='tight')
        plt.close()

        if verbose: print('Created: ',pfnm)


def make_tzpz_plot(verbose, runid, \
                   polygons_draw=False, polygons_vertices=None, polygons_color='green', \
                   outliers_show=True, outliers_color='red', outliers_label=False, \
                   make_pztz=False, plot_title = ''):

    '''
    Plot of true redshift vs. photometric redshift as a 2d histogram,
    with options to add polygons to define regions and/or show outliers as points.

    Plotting parameters
     polygons_draw     : True/False; if True polygons are drawn
     polygons_vertices : an array of x and y vertices for N polygonse
     polygons_color    : line color for polygons
     outliers_show     : True/False; if True outliers are shown as points
     outliers_color    : point color for outliers
     outliers_label    : True/False; if True outliers are labeled in a legend
     make_pztz         : swap axes and plot y=PHOT vs x=TRUE redshift
    '''

    if verbose:
        print(' ')
        print('Starting cmnn_analysis.make_tzpz_plot(), ',datetime.datetime.now())

    if os.path.isdir('output/run_'+runid+'/analysis') == False :
        os.system('mkdir output/run_'+runid+'/analysis')

    # Read in the data
    fnm = 'output/run_'+runid+'/zphot.cat'
    ztrue = np.loadtxt(fnm, dtype='float', usecols=(1))
    zphot = np.loadtxt(fnm, dtype='float', usecols=(2))
    Ncm = np.loadtxt(fnm, dtype='int', usecols=(4))

    # Initialize the plot
    fig = plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size':20})

    # Draw a line of y=x to guide the eye
    plt.plot([0.0,3.0],[0.0,3.0],color='black')

    # Draw polygons
    #  Example: if you want two boxes centered on (0.25,0.50) and (2.0,1.2):
    #  polygons_vertices = [ [ [0.2,0.3,0.3,0.2,0.2], [0.45,0.45,0.55,0.55,0.45] ], \
    #                        [ [1.9,2.1,2.1,1.9,1.9], [1.1,1.1,1.3,1.3,1.1] ] ]
    if polygons_draw == True:
        for v in polygons_vertices:
            plt.plot(v[0], v[1], color=polygons_color, lw=2, alpha=1)

    # Include all test galaxies with a measured zphot in the 2d histogram
    tx = np.where(zphot > 0.0)[0]
    if make_pztz == False:
        plt.hist2d(zphot[tx], ztrue[tx], bins=100, range=[[0.0,3.0],[0.0,3.0]], norm=LogNorm(clip=True), \
            cmin=1, cmap='Greys')
    if make_pztz == True:
        plt.hist2d(ztrue[tx], zphot[tx], bins=100, range=[[0.0,3.0],[0.0,3.0]], norm=LogNorm(clip=True), \
            cmin=1, cmap='Greys')
    del tx

    # Overplot the outliers as points
    if outliers_show:
        tmp_x = np.where((zphot > float(0.300)) & (zphot < float(3.0)))[0]
        tmp_ztrue = ztrue[tmp_x]
        tmp_zphot = zphot[tmp_x]
        tmp_dz = (tmp_ztrue - tmp_zphot) / (1.0 + tmp_zphot)
        q75, q25 = np.percentile(tmp_dz, [75 ,25])
        sigma = (q75 - q25) / float(1.349)
        threesigma = float(3.0) * sigma
        ox = np.where((np.fabs(tmp_dz) > float(0.0600)) & (np.fabs(tmp_dz) > threesigma))[0]
        if outliers_label:
            if make_pztz == False:
                plt.plot(tmp_zphot[ox], tmp_ztrue[ox], 'o', ms=2, alpha=0.4, color=outliers_color, \
                    markeredgewidth=0, label='outlier')
            if make_pztz == True:
                plt.plot(tmp_ztrue[ox], tmp_zphot[ox], 'o', ms=2, alpha=0.4, color=outliers_color, \
                    markeredgewidth=0, label='outlier')
        else:
            if make_pztz == False:
                plt.plot(tmp_zphot[ox], tmp_ztrue[ox], 'o', ms=2, alpha=0.4, color=outliers_color, \
                    markeredgewidth=0)
            if make_pztz == True:
                plt.plot(tmp_ztrue[ox], tmp_zphot[ox], 'o', ms=2, alpha=0.4, color=outliers_color, \
                    markeredgewidth=0)
        del tmp_x, tmp_ztrue, tmp_zphot, tmp_dz
        del q75, q25, sigma, threesigma, ox

    # Set axis parameters and write plot to file
    if make_pztz == False:
        plt.xlabel('Photometric Redshift')
        plt.ylabel('True Redshift')
    if make_pztz == True:
        plt.xlabel('True Redshift')
        plt.ylabel('Photometric Redshift')
    plt.xlim([0.0,3.0])
    plt.ylim([0.0,3.0])
    if outliers_label:
        plt.legend(loc='upper center', numpoints=1, markerscale=3, prop={'size':16}, labelspacing=0.5)

    if make_pztz == False:
        ofnm = 'output/run_'+runid+'/analysis/tzpz'
    if make_pztz == True:
        ofnm = 'output/run_'+runid+'/analysis/pztz'
        
    if plot_title != '':
        plt.title(plot_title)                 
        
    plt.savefig(ofnm,bbox_inches='tight')
    plt.close()
    if verbose: print('Wrote to: ',ofnm)


def make_hist_plots(verbose, runid):

    if verbose:
        print(' ')
        print('Starting cmnn_analysis.make_hist_plot(), ',datetime.datetime.now())

    if os.path.isdir('output/run_'+runid+'/analysis') == False :
        os.system('mkdir output/run_'+runid+'/analysis')

    # Read in the data
    fnm = 'output/run_'+runid+'/zphot.cat'
    ztrue = np.loadtxt(fnm, dtype='float', usecols=(1))
    zphot = np.loadtxt(fnm, dtype='float', usecols=(2))
    Ncm = np.loadtxt(fnm, dtype='float', usecols=(4))
    Ntrain = np.loadtxt(fnm, dtype='float', usecols=(5))
    ztrain = np.loadtxt('output/run_'+runid+'/train.cat', dtype='float', usecols=(1))

    # Histogram of redshifts: test true, test photoz, train z
    pfnm = 'output/run_'+runid+'/analysis/hist_z'
    fig = plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size':20})
    plt.hist(ztrain, density=True,bins=100,histtype='step',ls='solid',\
            lw=2,alpha=0.5,color='grey',label='train z')
    plt.hist(ztrue, density=True,bins=100,histtype='step',ls='solid',\
            lw=2,alpha=0.7,color='blue',label='test z true')
    plt.hist(zphot, density=True,bins=100,histtype='step',ls='solid',\
            lw=2,alpha=0.7,color='red',label='test photo-z')
    plt.xlabel('Redshift')
    plt.ylabel('Fraction of Galaxies')
    plt.xlim([0.0,3.0])
    plt.legend(loc='upper right', numpoints=1, markerscale=3, prop={'size':16}, labelspacing=0.5)
    plt.savefig(pfnm,bbox_inches='tight')
    plt.close()
    if verbose: print('Wrote to: ',pfnm)

    # Histogram of Ncm: number of training galaxies in the CMNN subset
    pfnm = 'output/run_'+runid+'/analysis/hist_ncm'
    fig = plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size':20})
    plt.hist(Ncm, bins=100,histtype='step',ls='solid',\
            lw=2,alpha=0.7,color='red')
    plt.xlabel('Size of CMNN Subset')
    plt.ylabel('Number of Test Galaxies')
    plt.savefig(pfnm,bbox_inches='tight')
    plt.close()
    if verbose: print('Wrote to: ',pfnm)

    pfnm = 'output/run_'+runid+'/analysis/hist_ntr'
    fig = plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size':20})
    plt.hist(Ntrain, bins=100,histtype='step',ls='solid',\
            lw=2,alpha=0.7,color='red')
    plt.xlabel('Size of Post-Cuts Training Set')
    plt.ylabel('Number of Test Galaxies')
    plt.savefig(pfnm,bbox_inches='tight')
    plt.close()
    if verbose: print('Wrote to: ',pfnm)


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
import datetime


def get_stats( in_zspec, in_zphot, zlow, zhi, thresh_COR ):

    ### Return the statistical measures for a given redshift bin bounded by zlow,zi.

    ### Inputs 
    ###   in_zspec   : list of spectroscopic or 'true' redshifts for ALL test galaxies (0.3 to 3.0)
    ###   in_zphot   : list of photometric redshift point estimates for ALL test galaxies (0.3 to 3.0)
    ###   zlow       : low-end redshift of bin
    ###   zhi        : high-end redshift of bin
    ###   thresh_COR : zspec-zphot threshold to define "catastrophic outlier rejection"

    ### Definitions used in this function.
    ###   dzo1pzp              : dz over 1 plus zphot, ( zspec - zphot ) / ( 1 + zphot )
    ###   catastrophic outlier : test galaxy with | zspec - zphot | > stats_COR
    ###   outlier              : test galaxy with (|dzo1pzp| > 3*IQRs) & (|dzo1pzp| > 0.06)
    ### The definition of an "outlier" comes from the SRD (which calls them 'catastrophic outliers'; ls.st/srd).
    ### For outliers, the IQRs used to identify them is calculated from all test galaxies in 0.3 < zphot < 3.0

    ### Definitions of the statistical measures.
    ###   meanz   : the mean zphot of galaxies in bin
    ###   fout    : fraction of outliers (see note below)
    ###   stdd    : standard deviation in dzo1pzp of all galaxies in bin
    ###   bias    : mean dzo1pzp of all galaxies in bin
    ###   IQR     : interquartile range of dzo1pzp
    ###   IQRstdd : stdandard deviation from the IQR ( = IQR / 1.349 )
    ###   IQRbias : bias of test galaxies in the IQR 
    ###   COR     : a catastrophic outlier-rejected statistical measure

    ### Outputs returned
    ###   meanz       : meanz
    ###   CORmeanz    : post-COR, the meanz
    ###   fout        : fout
    ###   stdd        : stdd
    ###   bias        : bias
    ###   IQR         : IQR
    ###   IQRstdd     : IQR stdd (robust standard deviatin)
    ###   IQRbias     : IQR bias (robust bias)
    ###   CORstdd     : post-COR stdd
    ###   CORbias     : post-COR bias
    ###   CORIQR      : post-COR IQR
    ###   CORIQRstdd  : post-COR IQR stdd
    ###   CORIQRbias  : post-COR IQR bias
    ###   estdd       : error in stdd
    ###   ebias       : error in bias
    ###   eIQR        : error in IQR
    ###   eIQRstdd    : error in IQR stdd
    ###   eIQRbias    : error in IQR bias
    ###   eCORstdd    : error in post-COR stdd
    ###   eCORbias    : error in post-COR bias
    ###   eCORIQR     : error in post-COR IQR
    ###   eCORIQRstdd : error in post-COR IQR stdd
    ###   eCORIQRbias : error in post-COR IQR bias

    ### Bootstrap resample with replacement to estimate errors for the statistical measures.
    ###   Nmc : number of times to repeat measurement
    Nmc  = int(1000)

    ### Calculate the IQRs for all test galaxies with a zphot in the range 0.3 < zphot < 3.0.
    ### Will need this to identify outliers using the SRD's definition of an outlier.
    tx         = np.where( ( in_zphot >= float(0.300) ) & ( in_zphot <= float(3.000) ) )[0]
    fr_zspec   = in_zspec[tx]
    fr_zphot   = in_zphot[tx]
    fr_dzo1pzp = ( fr_zspec - fr_zphot ) / ( float(1.0) + fr_zphot )
    q75, q25   = np.percentile( fr_dzo1pzp, [75 ,25] )
    fr_IQRs    = ( q75 - q25 ) / float(1.349)
    del tx, fr_zphot, fr_zspec, fr_dzo1pzp, q75, q25

    ### Identify all test galaxies in the requested bin.
    tx    = np.where( ( in_zphot > zlow ) & ( in_zphot <= zhi ) )[0]
    zspec = in_zspec[tx]
    zphot = in_zphot[tx]
    del tx

    ### Identify the subset of catastrophic outlier-rejected test galaxies in the requested bin
    tx       = np.where( ( in_zphot > zlow ) & ( in_zphot <= zhi ) & ( np.abs( in_zphot - in_zspec ) < thresh_COR ) )[0]
    CORzspec = in_zspec[tx]
    CORzphot = in_zphot[tx]
    del tx

    del in_zspec, in_zphot

    ### Calculate the mean zphot in the bin
    meanz    = np.mean( zphot )
    CORmeanz = np.mean( CORzphot )

    ### Define bin_dzo1pzp for use in all stats
    dzo1pzp    = ( zspec - zphot ) / ( float(1.0) + zphot )
    CORdzo1pzp = ( CORzspec - CORzphot ) / ( float(1.0) + CORzphot )

    ### Fraction of outliers as defined by the SRD
    tx   = np.where( ( np.fabs( dzo1pzp ) > float(0.0600) ) & ( np.fabs( dzo1pzp ) > ( float(3.0)*fr_IQRs) ) )[0]
    fout = float( len( tx ) ) / float( len( dzo1pzp ) )
    del tx

    ### Standard deviation
    stdd    = np.std( dzo1pzp )
    CORstdd = np.std( CORdzo1pzp )

    ### Bias
    bias    = np.mean( dzo1pzp )
    CORbias = np.mean( CORdzo1pzp )

    ### Intraquartile Range
    q75, q25 = np.percentile( dzo1pzp, [75 ,25] )
    IQR      = ( q75 - q25 )
    IQRstdd  = ( q75 - q25 ) / float(1.349)
    tx       = np.where( ( dzo1pzp > q25 ) & ( dzo1pzp < q75 ) )[0]
    IQRbias  = np.mean( dzo1pzp[tx] )
    del q75, q25, tx

    ### COR Intraquartile Range
    q75, q25   = np.percentile( CORdzo1pzp, [75 ,25] )
    CORIQR     = ( q75 - q25 )
    CORIQRstdd = ( q75 - q25 ) / float(1.349)
    tx         = np.where( ( CORdzo1pzp > q25 ) & ( CORdzo1pzp < q75 ) )[0]
    CORIQRbias = np.mean( CORdzo1pzp[tx] )
    del q75, q25, tx

    ### Now do the MC and calculate errors for all quantities
    vals_s       = np.zeros( Nmc, dtype='float')
    vals_b       = np.zeros( Nmc, dtype='float')
    vals_IQR     = np.zeros( Nmc, dtype='float')
    vals_IQRs    = np.zeros( Nmc, dtype='float')
    vals_IQRb    = np.zeros( Nmc, dtype='float')
    vals_CORs    = np.zeros( Nmc, dtype='float')
    vals_CORb    = np.zeros( Nmc, dtype='float')
    vals_CORIQR  = np.zeros( Nmc, dtype='float')
    vals_CORIQRs = np.zeros( Nmc, dtype='float')
    vals_CORIQRb = np.zeros( Nmc, dtype='float')
    for i in range(Nmc):
        tx = np.random.choice( len(dzo1pzp), size=len(dzo1pzp), replace=True, p=None )
        vals_s[i]    = np.std( dzo1pzp[tx] )
        vals_b[i]    = np.mean( dzo1pzp[tx] )
        q75, q25     = np.percentile( dzo1pzp[tx], [75 ,25] )
        vals_IQR[i]  = ( q75 - q25 )
        vals_IQRs[i] = ( q75 - q25 ) / float(1.349)
        temp         = dzo1pzp[tx]
        ttx          = np.where( ( temp > q25 ) & ( temp < q75 ) )[0]
        vals_IQRb[i] = np.mean( temp[ttx] )
        del tx, q75, q25, temp, ttx

        tx = np.random.choice( len(CORdzo1pzp), size=len(CORdzo1pzp), replace=True, p=None )
        vals_CORs[i]    = np.std( CORdzo1pzp[tx] )
        vals_CORb[i]    = np.mean( CORdzo1pzp[tx] )
        q75, q25        = np.percentile( CORdzo1pzp[tx], [75 ,25])
        vals_CORIQR[i]  = ( q75 - q25 )
        vals_CORIQRs[i] = ( q75 - q25 ) / float(1.349)
        CORtemp         = CORdzo1pzp[tx]
        ttx             = np.where( ( CORtemp > q25 ) & ( CORtemp < q75 ) )[0]
        vals_CORIQRb[i] = np.mean( CORtemp[ttx] )
        del tx, q75, q25, CORtemp, ttx

    estdd       = np.std( vals_s )
    ebias       = np.std( vals_b )
    eIQR        = np.std( vals_IQR )
    eIQRstdd    = np.std( vals_IQRs )
    eIQRbias    = np.std( vals_IQRb )
    eCORstdd    = np.std( vals_CORs )
    eCORbias    = np.std( vals_CORb )
    eCORIQR     = np.std( vals_CORIQR )
    eCORIQRstdd = np.std( vals_CORIQRs )
    eCORIQRbias = np.std( vals_CORIQRb )
    del vals_s, vals_b, vals_IQR, vals_IQRs, vals_IQRb
    del vals_CORs, vals_CORb, vals_CORIQR, vals_CORIQRs, vals_CORIQRb

    return meanz, CORmeanz, fout, \
        stdd, bias, IQR, IQRstdd, IQRbias, CORstdd, CORbias, CORIQR, CORIQRstdd, CORIQRbias, \
        estdd, ebias, eIQR, eIQRstdd, eIQRbias, eCORstdd, eCORbias, eCORIQR, eCORIQRstdd, eCORIQRbias


def make_stats_file(  verbose, runid, stats_COR, input_zbins=None ):

    ### Make a file containing the photo-z statistics in zphot bins.

    ### Input definitions 
    ###   thresh_COR  : zspec-zphot threshold to define "catastrophic outlier rejection"
    ###   input_zbins : an array of redshift bin centers [ [zlow_1,zhi_1], [zlow_2,zhi_2], ... [zlow_N,zhi_N] ]

    if verbose:
        print(' ')
        print('Starting cmnn_analysis.make_stats_file(), ',datetime.datetime.now())

    if os.path.isdir('output/run_'+runid+'/analysis') == False :
        os.system('mkdir output/run_'+runid+'/analysis')

    ### Setup the redshift bins, the default is overlapping bins
    if input_zbins == None:
        temp = []
        zlow = float(0.00)
        zhi  = float(0.30)
        temp.append( [zlow,zhi] )
        for z in range(18):
            zlow += float(0.15)
            zhi  += float(0.15)
            temp.append( [zlow,zhi] )
        zbins = np.asarray( temp, dtype='float' )
        del temp,zlow,zhi
    else:
        zbins = input_zbins

    ### Read in the data
    fnm = 'output/run_'+runid+'/zphot.cat'
    ztrue = np.loadtxt( fnm, dtype='float', usecols={1})
    zphot = np.loadtxt( fnm, dtype='float', usecols={2})

    ### Open the output file for writing
    ### Then loop over all zbins, obtain the statistical measures, and write them to file
    ofnm = 'output/run_'+runid+'/analysis/stats.dat'
    fout = open(ofnm,'w')
    fout.write('# zlow zhi meanz CORmeanz fout '+\
        'stdd bias IQR IQRstdd IQRbias CORstdd CORbias CORIQR CORIQRstdd CORIQRbias '+\
        'estdd ebias eIQR eIQRstdd eIQRbias eCORstdd eCORbias eCORIQR eCORIQRstdd eCORIQRbias \n')
    for z in range(len(zbins)-1):
        stats = get_stats( ztrue, zphot, zbins[z,0], zbins[z,1], stats_COR )
        fout.write('%6.4f %6.4f ' % (zbins[z,0], zbins[z,1]) )
        fout.write('%6.4f %6.4f %6.4f ' % (stats[0], stats[1], stats[2]) )
        fout.write('%6.4f %6.4f %6.4f %6.4f %6.4f ' % (stats[3], stats[4], stats[5], stats[6], stats[7]) )
        fout.write('%6.4f %6.4f %6.4f %6.4f %6.4f ' % (stats[8], stats[9], stats[10], stats[11], stats[12]) )
        fout.write('%6.4f %6.4f %6.4f %6.4f %6.4f ' % (stats[13], stats[14], stats[15], stats[16], stats[17]) )
        fout.write('%6.4f %6.4f %6.4f %6.4f %6.4f \n' % (stats[18], stats[19], stats[20], stats[21], stats[22]) )
    fout.close()
    if verbose: print('Wrote to: ',ofnm)


def make_stats_plots( verbose=True, runid=None, user_stats=None, show_SRD=True, show_binw=True, \
    multi_run_ids=None, multi_run_labels=None, multi_run_colors=['blue','orange','red','green','darkviolet'] ):

    ### Make plots for all the photo-z statistics for a given run.

    ### Input definitions 
    ###  verbose          : extra output to screen
    ###  runid            : the run to make plots for (ignored if multi_run_ids != None)
    ###  user_stats       : array listing which stats to create plots for, default is:
    ###                     ['fout','CORIQRstdd','CORIQRbias']
    ###  show_SRD         : True/False; if True, the SRD target values are shown as dashed horizontal lines
    ###  multi_run_ids    : array of multiple run ids to co-plot
    ###  multi_run_labels : array of legend labels that describe each run
    ###  multi_run_colors : array of color names to use for each run (five provided as defaults)

    if verbose:
        print( ' ' )
        print( 'Starting cmnn_analysis.make_stats_plots(), ', datetime.datetime.now() )

    if runid==None:
        if multi_run_ids==None:
            print( ' ' )
            print( 'Error in cmnn_analysis.make_stats_plots()' )
            print( 'Need to specify input values for at least one of: runid or multi_run_ids.')
            print( '  runid = ',runid)
            print( '  multi_run_ids = ',multi_run_ids)
            print( 'Exit.' )
            exit()

    ### Check that cmnn_analysis.make_stats_file has been run already
    if os.path.exists('output/run_'+runid+'/analysis/stats.dat') == False:
        print( ' ' )
        print( 'Error, file does not exist: output/run_'+runid+'/analysis/stats.dat')
        print( 'Need to run cmnn_analysis.make_stats_file() first.')
        print( 'Exit.' )
        exit()

    ### Check user input for multiple runs
    if multi_run_ids == None:
        del multi_run_ids,multi_run_labels
        multi_run_ids    = [runid]
        multi_run_labels = ['run '+runid]
    if multi_run_ids != None:
        if (len(multi_run_ids) != len(multi_run_labels)) | (len(multi_run_ids) > len(multi_run_colors)):
            print( ' ' )
            print( 'Error in cmnn_analysis.make_stats_plots()' )
            print( 'User-defined input regarding the multiple runs to plot is incompatible.' )
            print( 'Values are:' )
            print( '  len(multi_run_ids) = ', len(multi_run_ids) )
            print( '  len(multi_run_labels) = ', len(multi_run_labels) )
            print( '  len(multi_run_colors) = ', len(multi_run_colors) )
            print( 'But it is required that:' )
            print( '  len(multi_run_ids) = len(multi_run_labls)' )
            print( '  len(multi_run_ids) <= len(multi_run_colors)' )
            print( 'Exit.' )
            exit()
    del runid

    all_stats_names = np.asarray( [\
        'fout','stdd','bias',\
        'IQR','IQRstdd','IQRbias',\
        'CORstdd','CORbias','CORIQR',\
        'CORIQRstdd','CORIQRbias'], dtype='str' )
    all_stats_labels = np.asarray( [\
        'Fraction of Outliers','Standard Deviation','Bias',\
        'IQR','Robust Standard Deviation','Robust Bias',\
        'Standard Deviation (C.O.R.)','Bias (C.O.R)','IQR (C.O.R.)',\
        'Robust Standard Deviation (C.O.R.)','Robust Bias (C.O.R.)'], dtype='str' )
    all_stats_xcols  = np.asarray( [  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype='int' )
    all_stats_ycols  = np.asarray( [  4, 5, 6, 7, 8, 9,10,11,12,13,14], dtype='int' )
    all_stats_yecols = np.asarray( [-99,15,16,17,18,19,20,21,22,23,24], dtype='int' )

    ### Populate the stats and names arrays to use when plotting
    if user_stats == None:
        user_stats = ['fout','CORIQRstdd','CORIQRbias']
    x = []
    for stat in user_stats:
        tx = np.where( stat == all_stats_names )[0]
        if len(tx) == 1:
            x.append(tx[0])
        del tx
    sx = np.asarray( x, dtype='int' )
    del x
    stats_names  = all_stats_names[sx]
    stats_labels = all_stats_labels[sx]
    stats_xcols   = all_stats_xcols[sx]
    stats_ycols   = all_stats_ycols[sx]
    stats_yecols  = all_stats_yecols[sx]
    del sx,user_stats
    del all_stats_names,all_stats_labels,all_stats_xcols,all_stats_ycols,all_stats_yecols

    ### Loop over all possible statistical measures and make a plot
    for s,stat in enumerate(stats_names):

        ### Define which columns to be read from the stats file
        sxc = stats_xcols[s]
        syc = stats_ycols[s]

        ### Initialize the plot
        fig = plt.figure(figsize=(12,8))
        plt.rcParams.update({'font.size':20})

        ### Show the SRD target values, if desired
        if show_SRD == True:
            ### Fraction of Outliers, 10%
            if syc == 4:
                plt.plot( [0.3,3.0], [0.10,0.10], lw=2, alpha=1, ls='dashed', color='black', label='SRD')
            ### Standard Deviation, 0.02
            if ( syc == 5 ) | ( syc == 7 ) | ( syc == 8 ) | ( syc == 10 ) | ( syc == 12 ) | ( syc == 13 ) :
                plt.plot( [0.3,3.0], [0.02,0.02], lw=2, alpha=1, ls='dashed', color='black', label='SRD')
            ### Bias, +/- 0.003
            if ( syc == 6 ) | ( syc == 9 ) | ( syc == 11 ) | ( syc == 14 ):
                plt.plot( [0.3,3.0], [-0.003,-0.003], lw=2, alpha=1, ls='dashed', color='black', label='SRD')
                plt.plot( [0.3,3.0], [0.003,0.003], lw=2, alpha=1, ls='dashed', color='black')

        pfnm_suffix = ''
        for r,runid in enumerate(multi_run_ids):
            pfnm_suffix += '_'+runid
            ### Read in the statistical measures
            sfnm = 'output/run_'+runid+'/analysis/stats.dat'
            ## If we're showing the bins as horizontal bars
            if show_binw:
                binlo = np.loadtxt( sfnm, dtype='float', usecols={0} )
                binhi = np.loadtxt( sfnm, dtype='float', usecols={1} )
            xvals = np.loadtxt( sfnm, dtype='float', usecols={sxc} )
            yvals = np.loadtxt( sfnm, dtype='float', usecols={syc} )
            if stats_yecols[s] > 0:
                eyvals = np.loadtxt( sfnm, dtype='float', usecols={stats_yecols[s]} )
            ### Plot the values, with error bars if appropriate
            plt.plot( xvals, yvals, lw=3, alpha=0.75, color=multi_run_colors[r], label=multi_run_labels[r] )
            ## If we're showing the bins as horizontal bars
            if show_binw:
                for y,yv in enumerate(yvals):
                    plt.plot( [binlo[y],binhi[y]], [yv,yv], lw=1, alpha=0.5, color=multi_run_colors[r] )
                del binlo,binhi
            if stats_yecols[s] > 0:
                plt.errorbar( xvals, yvals, yerr=eyvals, fmt='none', elinewidth=3, capsize=3, capthick=3, ecolor=multi_run_colors[r] )
                del eyvals
            del sfnm,xvals,yvals

        ### Set axis parameters and write plot to file
        plt.xlabel( 'Photometric Redshift' )
        plt.ylabel( stats_labels[s] )
        legend=plt.legend(loc='best',numpoints=1,prop={'size':14},labelspacing=0.15) #,title=lgnd_title)
        # legend.get_title().set_fontsize('14') 

        if len(multi_run_ids) == 1:
            pfnm = 'output/run_'+multi_run_ids[0]+'/analysis/'+stat
        if len(multi_run_ids) > 1:
            pfnm = 'output/stats_plots/'+stat+pfnm_suffix
            if os.path.exists('output/stats_plots') == False:
                os.system('mkdir output/stats_plots')
        plt.savefig( pfnm, bbox_inches='tight') 

        if verbose: print('Created: ',pfnm)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def make_tzpz_plot( verbose, runid, \
    polygons_draw=False, polygons_vertices=None, polygons_color='green', \
    outliers_show=True, outliers_color='red', outliers_label=False):

    ### Plot of true redshift vs. photometric redshift as a 2d histogram,
    ###  with options to add polygons to define regions and/or show outliers as points.

    ### Plotting parameters
    ###  polygons_draw         =  True/False; if True polygons are drawn
    ###  polygons_vertices     =  an array of x and y vertices for N polygonse
    ###  polygons_color        =  line color for polygons
    ###  outliers_show         =  True/False; if True outliers are shown as points
    ###  outliers_color        =  point color for outliers
    ###  outliers_label        =  True/False; if True outliers are labeled in a legend

    if verbose:
        print(' ')
        print('Starting cmnn_analysis.make_tzpz_plot(), ',datetime.datetime.now())

    if os.path.isdir('output/run_'+runid+'/analysis') == False :
        os.system('mkdir output/run_'+runid+'/analysis')

    ### Read in the data
    fnm = 'output/run_'+runid+'/zphot.cat'
    ztrue = np.loadtxt( fnm, dtype='float', usecols={1})
    zphot = np.loadtxt( fnm, dtype='float', usecols={2})
    Ncm   = np.loadtxt( fnm, dtype='int', usecols={4})

    ### Initialize the plot
    fig = plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size':20})

    ### Draw a line of y=x to guide the eye
    plt.plot([0.0,3.0],[0.0,3.0],color='black')

    ### Draw polygons
    ###  Example: if you want two boxes centered on (0.25,0.50) and (2.0,1.2):
    ###  polygons_vertices = [ [ [0.2,0.3,0.3,0.2,0.2], [0.45,0.45,0.55,0.55,0.45] ], \
    ###                        [ [1.9,2.1,2.1,1.9,1.9], [1.1,1.1,1.3,1.3,1.1] ] ]
    if polygons_draw == True:
        for v in polygons_vertices:
            plt.plot( v[0], v[1], color=polygons_color, lw=2, alpha=1)

    ### Include all test galaxies with a measured zphot in the 2d histogram
    tx = np.where( zphot > 0.0 )[0]
    plt.hist2d( zphot[tx], ztrue[tx], bins=100, range=[[0.0,3.0],[0.0,3.0]], norm=LogNorm(clip=True), \
        cmin=1, cmap='Greys')
    del tx

    ### Overplot the outliers as points
    if outliers_show:
        tmp_x      = np.where( (zphot > float(0.300)) & (zphot < float(3.0)) )[0]
        tmp_ztrue  = ztrue[tmp_x]
        tmp_zphot  = zphot[tmp_x]
        tmp_dz     = ( tmp_ztrue - tmp_zphot ) / ( 1.0 + tmp_zphot )
        q75, q25   = np.percentile( tmp_dz, [75 ,25])
        sigma      = ( q75 - q25 ) / float(1.349)
        threesigma = float(3.0) * sigma
        ox = np.where( ( np.fabs( tmp_dz ) > float(0.0600) ) & ( np.fabs( tmp_dz ) > threesigma ) )[0]
        if outliers_label:
            plt.plot( tmp_zphot[ox], tmp_ztrue[ox], 'o', ms=2, alpha=0.4, color=outliers_color, \
                markeredgewidth=0, label='outlier')
        else:
            plt.plot( tmp_zphot[ox], tmp_ztrue[ox], 'o', ms=2, alpha=0.4, color=outliers_color, \
                markeredgewidth=0)
        del tmp_x, tmp_ztrue, tmp_zphot, tmp_dz
        del q75, q25, sigma, threesigma, ox

    ### Set axis parameters and write plot to file
    plt.xlabel('Photometric Redshift')
    plt.ylabel('True Redshift')
    plt.xlim([0.0,3.0])
    plt.ylim([0.0,3.0])
    if outliers_label:
        plt.legend( loc='upper center', numpoints=1, markerscale=3, prop={'size':16}, labelspacing=0.5)

    ofnm = 'output/run_'+runid+'/analysis/tzpz'
    plt.savefig(ofnm,bbox_inches='tight')
    if verbose: print('Wrote to: ',ofnm)



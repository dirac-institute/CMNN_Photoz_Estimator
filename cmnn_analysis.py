import os
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize


def plot_tzpz_hist2d( input_zphot_catalog, output_plot_filename):

    # This function will make a simple plot of true redshift vs. photometric redshift as a 2d histogram

    # Input definitions 
    #  input_zphot_catalog   =  name of the data file containing true and photometric redshifts
    #  output_plot_filename  =  name for the file to write the plot to

    print('Starting: plot_tzpz_hist2d')

    # Read in the data
    ztrue = np.loadtxt( input_zphot_catalog, dtype='float', usecols={1})
    zphot = np.loadtxt( input_zphot_catalog, dtype='float', usecols={2})

    # Initialize the plot
    fig = plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size':20})

    # Draw a line of y=x to guide the eye
    plt.plot([0.0,3.0],[0.0,3.0],color='black')

    # Include all test galaxies with a measured zphot in the 2d histogram
    tx = np.where( zphot > 0.0 )[0]
    plt.hist2d( zphot[tx], ztrue[tx], bins=100, range=[[0.0,3.0],[0.0,3.0]], norm=LogNorm(clip=True), cmin=1, cmap='Greys')
    del tx

    # Set axis parameters and write plot to file
    plt.xlabel('Photometric Redshift')
    plt.ylabel('True Redshift')
    plt.xlim([0.0,3.0])
    plt.ylim([0.0,3.0])
    plt.savefig(output_plot_filename,bbox_inches='tight')

    print('Wrote to: ',output_plot_filename)
    print('Finished: plot_tzpz_hist2d')


def plot_tzpz_hist2d_fancy( input_zphot_catalog, output_plot_filename, \
    polygons_draw=False, polygons_vertices=None, polygons_color='green', \
    outliers_show=False, outliers_color='red', outliers_label=False):

    # This function will make a fancy plot of true redshift vs. photometric redshift as a 2d histogram
    #  with options to add polygons to define regions and/or show outliers as points.

    # Input definitions 
    #  input_zphot_catalog   =  name of the data file containing true and photometric redshifts
    #  output_plot_filename  =  name for the file to write the plot to
    #  polygons_draw         =  True/False; if True polygons are drawn
    #  polygons_vertices     =  an array of x and y vertices for N polygonse
    #  polygons_color        =  line color for polygons
    #  outliers_show         =  True/False; if True outliers are shown as points
    #  outliers_color        =  point color for outliers
    #  outliers_label        =  True/False; if True outliers are labeled in a legend

    print('Starting: plot_tzpz_hist2d_fancy')

    # Read in the data
    ztrue = np.loadtxt( input_zphot_catalog, dtype='float', usecols={1})
    zphot = np.loadtxt( input_zphot_catalog, dtype='float', usecols={2})

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
            plt.plot( v[0], v[1], color=polygons_color, lw=2, alpha=1)

    # Include all test galaxies with a measured zphot in the 2d histogram
    tx = np.where( zphot > 0.0 )[0]
    plt.hist2d( zphot[tx], ztrue[tx], bins=100, range=[[0.0,3.0],[0.0,3.0]], norm=LogNorm(clip=True), cmin=1, cmap='Greys')
    del tx

    # Overplot the outliers as red points
    tmp_x      = np.where( (zphot > 0.3) & (zphot < 3.0) )[0]
    tmp_ztrue  = ztrue[tmp_x]
    tmp_zphot  = zphot[tmp_x]
    tmp_dz     = ( tmp_ztrue - tmp_zphot ) / ( 1.0 + tmp_zphot )
    q75, q25   = np.percentile( tmp_dz, [75 ,25])
    sigma      = ( q75 - q25 ) / 1.349
    threesigma = 3.0 * sigma
    ox = np.where( ( np.fabs( tmp_dz ) > 0.06 ) & ( np.fabs( tmp_dz ) > threesigma ) )[0]
    if outliers_label:
        plt.plot( tmp_zphot[ox], tmp_ztrue[ox], 'o', ms=2, alpha=0.4, color=outliers_color, markeredgewidth=0, label='outlier')
    else:
        plt.plot( tmp_zphot[ox], tmp_ztrue[ox], 'o', ms=2, alpha=0.4, color=outliers_color, markeredgewidth=0)
    del tmp_x, tmp_ztrue, tmp_zphot, tmp_dz
    del q75, q25, sigma, threesigma, ox

    # Set axis parameters and write plot to file
    plt.xlabel('Photometric Redshift')
    plt.ylabel('True Redshift')
    plt.xlim([0.0,3.0])
    plt.ylim([0.0,3.0])
    if outliers_label:
        plt.legend( loc='upper center', numpoints=1, markerscale=3, prop={'size':16}, labelspacing=0.5)
    plt.savefig(output_plot_filename,bbox_inches='tight')

    print('Wrote to: ',output_plot_filename)
    print('Finished: plot_tzpz_hist2d_fancy')


def get_stats( in_zspec, in_zphot, zlow, zhi ):

    # This function will return the statistical measures for a given redshift bin bounded by zlow,zi.

    # Input definitions 
    #  in_zspec  =  list of spectroscopic or 'true' redshifts for test galaxies
    #  in_zphot  =  list of photometric redshift point estimates for test galaxies
    #  zlow      =  low-end redshift of bin
    #  zhi       =  high-end redshift of bin

    # Definitions used in this function.
    #  dzo1pzp              : dz over 1 plus zphot, ( zspec - zphot ) / ( 1 + zphot )
    #  catastrophic outlier : test galaxy with | zspec - zphot | > 2
    #  outlier              : test galaxy with (|dzo1pzp| > 3*IQRs) & (|dzo1pzp| > 0.06)
    # The definition of an "outlier" comes from the SRD (which calls them 'catastrophic outliers'; ls.st/srd).
    # For outliers, the IQRs used to identify them is calculated from all test galaxies in 0.3 < zphot < 3.0

    # Definitions of the statistical measures.
    #  meanz    = the mean zphot of galaxies in bin
    #  fout     = fraction of outliers (see note below)
    #  stdd     = standard deviation in dzo1pzp of all galaxies in bin
    #  bias     = mean dzo1pzp of all galaxies in bin
    #  IQR      = interquartile range of dzo1pzp
    #  IQRstdd  = stdandard deviation from the IQR ( = IQR / 1.349 )
    #  IQRbias  = bias of test galaxies in the IQR 
    #  COR      = a catastrophic outlier-rejected statistical measure

    # Bootstrap resample with replacement to estimate errors for the statistical measures.
    #  Nmc = number of times to repeat measurement
    Nmc  = 1000

    # Calculate the IQRs for all test galaxies with a zphot in the range 0.3 < zphot < 3.0.
    # Will need this to identify outliers using the SRD's definition of an outlier.
    tx         = np.where( ( in_zphot >= 0.3) & ( in_zphot <= 3.0) )[0]
    fr_zspec   = in_zspec[tx]
    fr_zphot   = in_zphot[tx]
    fr_dzo1pzp = ( fr_zspec - fr_zphot ) / ( 1.0 + fr_zphot )
    q75, q25   = np.percentile( fr_dzo1pzp, [75 ,25] )
    fr_IQRs    = ( q75 - q25 ) / 1.349
    del tx, fr_zphot, fr_zspec, fr_dzo1pzp, q75, q25

    # Identify all test galaxies in the requested bin.
    tx    = np.where( ( in_zphot > zlow ) & ( in_zphot <= zhi ) )[0]
    zspec = in_zspec[tx]
    zphot = in_zphot[tx]
    del tx

    # Identify the subset of catastrophic outlier-rejected test galaxies in the requested bin
    tx       = np.where( ( in_zphot > zlow ) & ( in_zphot <= zhi ) & ( np.abs( in_zphot - in_zspec ) < 2.0 ) )[0]
    CORzspec = in_zspec[tx]
    CORzphot = in_zphot[tx]
    del tx

    del in_zspec, in_zphot

    # Calculate the mean zphot in the bin
    meanz    = np.mean( zphot )
    CORmeanz = np.mean( CORzphot )

    # Define bin_dzo1pzp for use in all stats
    dzo1pzp    = ( zspec - zphot ) / ( 1.0 + zphot )
    CORdzo1pzp = ( CORzspec - CORzphot ) / ( 1.0 + CORzphot )

    # Fraction of outliers as defined by the SRD
    tx   = np.where( ( np.fabs( dzo1pzp ) > 0.06 ) & ( np.fabs( dzo1pzp ) > (3.0*fr_IQRs) ) )[0]
    fout = float( len( tx ) ) / float( len( dzo1pzp ) )
    del tx

    # Standard deviation
    stdd    = np.std( dzo1pzp )
    CORstdd = np.std( CORdzo1pzp )

    # Bias
    bias    = np.mean( dzo1pzp )
    CORbias = np.mean( CORdzo1pzp )

    # Intraquartile Range
    q75, q25 = np.percentile( dzo1pzp, [75 ,25] )
    IQR      = ( q75 - q25 )
    IQRstdd  = ( q75 - q25 ) / 1.349
    tx       = np.where( ( dzo1pzp > q25 ) & ( dzo1pzp < q75 ) )[0]
    IQRbias  = np.mean( dzo1pzp[tx] )
    del q75, q25, tx

    # COR Intraquartile Range
    q75, q25   = np.percentile( CORdzo1pzp, [75 ,25] )
    CORIQR     = ( q75 - q25 )
    CORIQRstdd = ( q75 - q25 ) / 1.349
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
        vals_IQRs[i] = ( q75 - q25 ) / 1.349
        temp         = dzo1pzp[tx]
        ttx          = np.where( ( temp > q25 ) & ( temp < q75 ) )[0]
        vals_IQRb[i] = np.mean( temp[ttx] )
        del tx, q75, q25, temp, ttx

        tx = np.random.choice( len(CORdzo1pzp), size=len(CORdzo1pzp), replace=True, p=None )
        vals_CORs[i]    = np.std( CORdzo1pzp[tx] )
        vals_CORb[i]    = np.mean( CORdzo1pzp[tx] )
        q75, q25        = np.percentile( CORdzo1pzp[tx], [75 ,25])
        vals_CORIQR[i]  = ( q75 - q25 )
        vals_CORIQRs[i] = ( q75 - q25 ) / 1.349
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



def make_stats_file( input_zphot_catalog, output_stats_filename, input_zbins=None ):

    # This function will make a file containing the photo-z statistics in zphot bins.

    # Input definitions 
    #  input_zphot_catalog   =  name of the data file containing true and photometric redshifts
    #  output_plot_filename  =  name for the file to write the statistical measures to

    print('Starting: make_stats_file')

    # Read in the data
    ztrue = np.loadtxt( input_zphot_catalog, dtype='float', usecols={1})
    zphot = np.loadtxt( input_zphot_catalog, dtype='float', usecols={2})

    # Setup the redshift bins
    if input_zbins == None:
        zbins = np.arange( 11, dtype='float' ) * 0.30
    else:
        zbins = input_zbins

    # Open the output file for writing
    fout = open(output_stats_filename,'w')
    fout.write('# zlow zhi meanz CORmeanz fout '+\
        'stdd bias IQR IQRstdd IQRbias CORstdd CORbias CORIQR CORIQRstdd CORIQRbias '+\
        'estdd ebias eIQR eIQRstdd eIQRbias eCORstdd eCORbias eCORIQR eCORIQRstdd eCORIQRbias \n')

    # Loop over all zbins, obtain the statistical measures, and write them to file
    for z in range(len(zbins)-1):
        stats = get_stats( ztrue, zphot, zbins[z], zbins[z+1] )
        fout.write('%4.3f %4.3f ' % (zbins[z], zbins[z+1]) )
        fout.write('%6.4f %6.4f %6.4f ' % (stats[0], stats[1], stats[2]) )
        fout.write('%6.4f %6.4f %6.4f %6.4f %6.4f ' % (stats[3], stats[4], stats[5], stats[6], stats[7]) )
        fout.write('%6.4f %6.4f %6.4f %6.4f %6.4f ' % (stats[8], stats[9], stats[10], stats[11], stats[12]) )
        fout.write('%6.4f %6.4f %6.4f %6.4f %6.4f ' % (stats[13], stats[14], stats[15], stats[16], stats[17]) )
        fout.write('%6.4f %6.4f %6.4f %6.4f %6.4f \n' % (stats[18], stats[19], stats[20], stats[21], stats[22]) )
    fout.close()

    print('Wrote to: ',output_stats_filename)
    print('Finished: make_stats_file')


def plot_stat( input_stat_filename, use_stats_and_names=None, show_SRD=False ):

    # This function will make plots for all the photo-z statistics

    # Input definitions 
    #  input_stat_filename  =  the name of the statistics file to be used for the plot
    #  show_SRD             =  True/False; if True, the SRD target values are shown as dashed horizontal lines

    print('Starting: plot_stat')

    # Define which statistics to plot, and what the axis should be named
    if use_stats_and_names == None:
        stats = np.asarray( ['fout','stdd','bias','IQR','IQRstdd','IQRbias',\
            'CORstdd','CORbias','CORIQR','CORIQRstdd','CORIQRbias'], dtype='str' )
        names = np.asarray( ['Fraction of Outliers','Standard Deviation','Bias','IQR','IQR Standard Deviation','IQR Bias',\
            'COR Standard Deviation','COR Bias','COR IQR','COR IQR Standard Deviation','COR IQR Bias'], dtype='str' )
    else:
        stats = np.asarray( use_stats_and_names[0], dtype='str' )
        names = np.asarray( use_stats_and_names[1], dtype='str' )

    # Loop over all possible statistical measures and make a plot
    for s,stat in enumerate(stats):

        # Initialize the plot
        fig = plt.figure(figsize=(12,8))
        plt.rcParams.update({'font.size':20})

        # Show the SRD target values, if desired
        if show_SRD == True:
            if s==0:
                plt.axhline( 0.10, lw=2, alpha=1, ls='dashed', color='green')
            if ( s == 1 ) | ( s == 4 ) | ( s == 6 ) | ( s == 9 ):
                plt.axhline( 0.02, lw=2, alpha=1, ls='dashed', color='green')
            if ( s == 2 ) | ( s == 5 ) | ( s == 7 ) | ( s == 10 ):
                plt.axhline( -0.003, lw=2, alpha=1, ls='dashed', color='green')
                plt.axhline( 0.003, lw=2, alpha=1, ls='dashed', color='green')

        # Read the correct columns for this statistic
        if stat[0:4] == 'COR':
            xcol = 3
        else:
            xcol = 2
        if stat == 'fout':
            ycol = 4
            eycol = -99
        else:
            ycol = s+4
            eycol = s+14
        xvals = np.loadtxt( input_stat_filename, dtype='float', usecols={xcol} )
        yvals = np.loadtxt( input_stat_filename, dtype='float', usecols={ycol} )
        if eycol > 0:
            eyvals = np.loadtxt( input_stat_filename, dtype='float', usecols={eycol} )

        # Plot the values, with error bars if appropriate
        plt.plot( xvals, yvals, lw=3, alpha=0.75, color='blue' )
        if eycol > 0:
            plt.errorbar( xvals, yvals, yerr=eyvals, fmt='none', elinewidth=3, capsize=3, capthick=3, ecolor='blue' )

        # Set axis parameters and write plot to file
        plt.xlabel('Photometric Redshift')
        plt.ylabel(names[s])
        plt.savefig('analysis/'+stat,bbox_inches='tight')

    print('Finished: plot_stat')



if __name__ == '__main__':

    # Make a simple plot of the true vs. photometric redshifts
    plot_tzpz_hist2d( 'zphot.dat', 'analysis/plot_tzpz' )

    # Make a fancier plot of the true vs. photometric redshifts
    vertices = [ [ [0.4,0.6,0.6,0.4,0.4], [0.15,0.15,0.4,0.4,0.15] ], [ [0.9,1.1,1.1,0.9,0.9], [1.15,1.15,1.85,1.85,1.15] ] ]
    plot_tzpz_hist2d_fancy( 'zphot.dat', 'analysis/plot_tzpz_fancy', \
    polygons_draw=True, polygons_vertices=vertices, polygons_color='green', \
    outliers_show=True, outliers_color='red', outliers_label=True)

    # Make the file of statistical measures as a function of photo-z, in photo-z bins
    make_stats_file( 'zphot.dat', 'analysis/zphot_stats.dat' )

    # Plot the desired statistics as a function of photo-z bin
    desired_stats_to_plot = [ ['fout','CORIQRstdd','CORIQRbias'], ['Fraction of Outliers','COR IQR Standard Deviation','COR IQR Bias'] ]
    plot_stat( 'analysis/zphot_stats.dat', use_stats_and_names=desired_stats_to_plot, show_SRD=True)



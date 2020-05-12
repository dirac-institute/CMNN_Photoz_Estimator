# CMNN_Photoz_Estimator
A Python3 implementation of the Color-Matched Nearest-Neighbors photometric redshift estimator.<br>
Melissa L. Graham, 2020. <br>


## An Overview of the CMNN Estimator
The color-matched nearest-neighbors (CMNN) photometric redshift estimator is described thoroughly in Graham et al. (2018, 2020).

**Simulated Test and Training Sets:**
The CMNN Estimator uses a _training set_ with known (spectroscopic) redshifts to estimate photometric redshifts for a _test set_ of galaxies.
In order to simulate a test and training set of galaxies with photometry that represents any desired LSST observing strategy, the same mock galaxy catalog as Graham et al. (2018, 2020) is used here.
As in those works, the observed apparent magnitudes and uncertainties for all galaxies are calculated from the true catalog apparent magnitudes, based on the predicted $5{\sigma}$ limiting magnitude depths for any given LSST observing strategy.
The photometric error, $\sigma_{\rm rand}$, is given in Section 3.2.1 of Ivezić et al. (2019): $\sigma^2_{\rm rand} = (0.04-\gamma)x + \gamma x^2$, where $x=10^{0.4(m-m_5)}$, $m_5$ is the $5\sigma$ limiting magnitude, $m$ is the magnitude of the galaxy, and for the LSST optical filters the values for $\gamma$, which sets the impact of, e.g., sky brightness, are $0.037$, $0.038$, $0.039$, $0.039$, $0.04$, and $0.04$ for filters {\it ugrizy}, respectively.
A random value drawn from a normal distribution with a standard deviation equal to the photometric error for each galaxy is added to the true catalog magnitude to generate observed apparent magnitudes.
For example, the photometric error of a galaxy with a true catalog magnitude of $i=24.5$ mag would be $\sigma_{\rm rand} = 0.026$; a random normal draw of $0.054$ would lead to an observed apparent magnitude of $i=24.55$ mag.
Observed apparent colors are calculated from the observed apparent magnitudes, and the error in the color is the photometric errors added in quadrature.

**Estimating Photometric Redshifts:**
For each galaxy in the test set, the estimator identifies a color-matched subset of training galaxies by calculating the Mahalanobis distance in color-space between the test galaxy and all training galaxies:<br>
$D_M = \sum_{\rm 1}^{N_{\rm colors}} \frac{( c_{\rm train} - c_{\rm test} )^2}{ (\delta c_{\rm test})^2}$<br>
where $c$ is the color of the test- or training-set galaxy, $\delta c_{\rm test}$ is the uncertainty in the test galaxy's color, and $N_{\rm color}$ is the number of colors measured for both the test- and training-set galaxy. 
A threshold value is then applied to all training-set galaxies to identify those which are well-matched in color, as defined by the percent point function (PPF): e.g., if the number of degrees of freedom $N_{\rm color}=5$, PPF$=68\%$ of all training galaxies consistent with the test galaxy will have $D_M < 5.86$.
A training galaxy is then selected by one of three modes from this subset of color-matched nearest-neighbors, and its redshift is used as the test-set galaxy's photometric redshift.
The standard deviation in redshifts of this subset of training galaxies is used as the uncertainty in the photo-$z$ estimate.
The three modes of selection are to choose randomly, to choose the galaxy with the lowest $D_M$, or to weight the random selection by the inverse of the $D_M$.

**Summary:**
The CMNN Estimator thus provides photometric redshifts with an accuracy that is directly dependent on (and only on) the quality of the simulated observed photometry which, as described above, depends directly and only on the $5{\sigma}$ limiting magnitudes.
This makes it an ideal tool for assessing the impact on photo-$z$ results from different proposed LSST observing strategies that build image depth over time in different ways, or result in different coadded depths.


## Capabilitites and Boundaries
This repository allows the user to specify $5{\sigma}$ limiting magnitude depths for the 6 Rubin Observatory filters _ugrizy_, sizes and magnitude cut-offs of the test and training sets, and parameters for the CMNN Estimator. The codes will generate test and training sets with appropriate simulated apparent magnitudes and errors, estimate photometric redshifts for test set galaxies, and perform a statistical analysis which generates diagnostic plots. All of these steps are described in Section 2 of Graham et al. (2020). The codes in this repository are designed to do all of these steps automatically for a single set of user input (a single "run") from one command-line call to cmnn\_run.py.

The provided galaxy catalog, "LSST\_galaxy\_catalog\_i25p3.dat", is equivalent to the mock catalog used by Graham et al. (2018,2020), but limited to galaxies with true apparent magnitudes $i\leq25.3$ mag. Furthermore, the test and training sets may not include galaxies with simulated observed magnitudes $i>25$ mag. This matches the $i\leq25$ mag "gold sample" defined by the Science Requirement Document (ls.st/lpm-17), and matches many of the photo-$z$ simulations in Graham et al. (2018,2020). For access to a deeper catalog contact the author (and then in cmnn\_run.run, change the values of 'mcut\_max').

Comparing the statistical results of multiple runs can be done with optional configurations of the make\_stats\_plots module in cmnn\_analysis.py, but this is not automatic.

The codes in this repository rely on the following packages: os, datetime, matplotlib, numpy, and scipy. 


## Examples: Photo-z results for a given year of the baseline observing strategy.

First, obtain the 5$\sigma$ limiting magnitudes for years 1, 3, 5, and 10.
```
path/CMNN_Photoz_Estimator: python
>>> import cmnn_tools
>>> for yr in [1,3,6,10]"
...     print( cmnn_tools.convert_year_to_depths_baseline( yr )
...
[ 24.83523503  26.12886248  26.28102228  25.58102228  24.80514998  23.60514998]
[ 25.4316366   26.72526405  26.87742385  26.17742385  25.40155155  24.20155155]
[ 25.8079241   27.10155155  27.25371134  26.55371134  25.77783904  24.57783904]
[ 26.08523503  27.37886248  27.53102228  26.83102228  26.05514998  24.85514998]
```

Next, run the CMNN Estimator for each of these desired years, passing the 5$\sigma$ limiting magnitudes to the test set input parameters. The magnitude cuts are set also to the 5$\sigma$ detection limit, with the exception of i-band which should always be set to 25 mag (or brighter). The training set limits will be kept to the default (the 10-year baseline).
```
python cmnn_run.py --verbose True --runid year1 --test_m5 24.84 26.13 26.28 25.58 24.81 23.61 --test_mcut 24.84 26.13 26.28 25.00 24.81 23.61
python cmnn_run.py --verbose True --runid year3 --test_m5 25.43 26.73 26.88 26.18 25.40 24.20 --test_mcut 25.43 26.73 26.88 25.00 25.40 24.20
python cmnn_run.py --verbose True --runid year6 --test_m5 25.81 27.10 27.25 26.55 25.78 24.58 --test_mcut 25.81 27.10 27.25 25.00 25.78 24.58
python cmnn_run.py --verbose True --runid year10 --test_m5 26.09 27.38 27.53 26.83 26.06 24.86 --test_mcut 26.09 27.38 27.53 25.00 26.06 24.86
```

Once the codes are finished, view the plots in the individual output directories:
```
path/CMNN_Photoz_Estimator: open output/run_year1/*.png
```

Make plots comparing the statististical measures for all years and view them.
```
path/CMNN_Photoz_Estimator: python
>>> import cmnn_analysis
>>> cmnn_analysis.make_stats_plots( multi_run_ids=['year1','year3','year6','year10'], multi_run_labels=['Year 1','Year 3','Year 6','Year 10'] )
path/CMNN_Photoz_Estimator: open output/stats_plots/*_year1_year3_year6_year10.png
```


## Module Summaries

### cmnn_run.main
Called from the command line.<br>
Parses the command line input. Checks that input values are good. Creates an output directory. Writes a file containing the input values to the output directory. Passes input to cmnn_run.run. 
This is the only code which can be called from the command line, and the only code which validates user-specified parameters.<br>

| Inputs | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| verbose | bool | True | if True, prints more intermediate information to the screen |
| runid | str | '1' | unique run identifier for labeling the output files |
| clobber | bool | False | if True, overwrites any existing output for this runid |
| test\_m5 | float[6] | [26.1, 27.4, 27.5, 26.8, 26.1, 24.9]$^{*}$ | the 5-sigma magnitude limits (depths) to apply to the test-set galaxies |
| train\_m5 | float[6] | [26.1, 27.4, 27.5, 26.8, 26.1, 24.9]$^{*}$ | the 5-sigma magnitude limits (depths) to apply to the training-set galaxies |
| test\_mcut | float[6] | [26.1, 27.4, 27.5, 25.0, 26.1, 24.9]$^{*}$ | a magnitude cut-off to apply to the test-set galaxies |
| train\_mcut | float[6] | [26.1, 27.4, 27.5, 25.0, 26.1, 24.9]$^{*}$ | a magnitude cut-off to apply to the training-set galaxies |
| force\_idet | bool | True | force detection in i-band for all test and train galaxies |
| test\_N | int | 40000 | number of test-set galaxies | 
| train\_N | int | 200000 | number of training-set galaxies | 
| cmnn\_minNc | int | 3 | minimum number of colors required for inclusion in catalog |
| cmnn\_minNN | int | 10 | forced minimum number of training-set galaxies in the CMNN subset |
| cmnn\_ppf | float | 0.68 | the percent point function that defines the Mahalanobis distance threshold of the CMNN |
| cmnn\_rsel | int | 2 | mode of random selection of a training-set galaxy from the CMNN subset (0 = random; 1 = nearest neighbor; 2 = random weighted by inverse of Mahalanobis distance) |
| cmnn\_ppmag | bool | False | apply a "pseudo-prior" to the training set based on the test-set's i-band magnitude (can only be True if force\_idet=True) |
| cmnn\_ppclr | bool | True | apply a "pseudo-prior" to the training set based on the test-set's g-r and r-i color |
| stats\_COR | float | 1.5 | catastrophic outlier rejection; reject galaxies with $(z_{true}-z_{phot})/(1+z_{phot}) >$ this value from the statistical measures of standard deviation and bias |

$^{*}$The default limiting magnitudes are the 10-year 5$\sigma$ limiting magnitude depths for a baseline observing strategy which accumulates 56, 80, 184, 184, 160, 160 visits in filters _ugrizy_ (both m5 and mcut, except for i-band where mcut=25).

| Exit Conditions |
| :-- |
| directory "output/run\__runid_/" exists and clobber=False |
| any of the magnitude arrays do not contain 6 floats |
| any of the 5$\sigma$ limits are less than [23.9, 25.0, 24.7, 24.0, 23.3, 22.1] (single-visit depths) |
| any of the 5$\sigma$ limits are greater than [29.0, 29.0, 29.0, 29.0, 29.0, 29.0] (arbitrary maximums) |
| any of the magnitude cuts are less than [17.0, 17.0, 17.0, 17.0, 17.0, 17.0] (nearing saturation) |
| any of the magnitude cuts are greater than [29.0, 29.0, 29.0, 25.0, 29.0, 29.0] (arbitrary, and $i<25$ mag) |
| cmnn\_ppmag is True and force\_idet is False (must require i-band detections to use magnitude psuedo-prior) |
| test\_N is $\leq0$ or $>100000$ |
| train\_N is $\leq0$ or $>1000000$ |
| stats\_COR is $\leq0$ |

| Outputs | Description |
| :------------- | :------------- |
| output/run\__runid_/ | directory for which _runid_ is a user-specified string |
| output/run\__runid_/inputs.txt | file listing in the values passed to cmnn\_run.run (user inputs or defaults) |
| output/run\__runid_/timestamps.dat | file in output directory listing the time at the start of each module |

### cmnn_run.run
Called by cmnn\_run.main.<br>
Passes user-specified input to the following modules, in order:<br>
cmnn_catalog.make_test_and_train<br>
cmnn_catalog.make_plots<br>
cmnn_photoz.make_zphot<br>
cmnn_analysis.make_tzpz_plot<br>
cmnn_analysis.make_stats_file<br>
cmnn_analysis.make_stats_plots

**Outputs**: None from this module alone, but it does contribute to timestamps.dat.

### cmnn_catalog.make_test_and_train
Called by cmnn\_run.run.<br>
Simulates observed apparent magnitudes for test and training sets based on the user-specified 5$\sigma$ limiting magnitudes (test\_m5 and train\_m5), and applies the user-specified cuts (test\_mcut, train\_mcut). Chooses randomly from the full mock catalog to create user-specified number of test and training set galaxies.

**Exit Condition:** If the user input sizes for the test and training sets are larger than the catalog can support given the depths and cuts, and error message will be returned and the code will exit without writing.

**Outputs:** Writes data files for test and training sets to the output directory (test.cat and train.cat).

### cmnn_catalog.make_plots
Called by cmnn\_run.run.<br>
Generates histograms of the redshifts and apparent magnitudes for the test and training sets.

**Outputs:** Saves plots to the output directory (cat_hist_ztrue.png and cat_hist_mag.png).

### cmnn_photoz.make_zphot
Called by cmnn\_run.run.<br>
Esimates photometric redshifts using the test and training sets of a given run.

**Outputs:** Writes test-set photo-z file to the output directory (zphot.cat).

### cmnn_photoz.return_photoz
Called by cmnn\_zphot.make_zphot.<br>
For a given test-set galaxy and a given training-set of galaxies, returns the estimated photometric redshift. 

### cmnn_analysis.make_tzpz_plot
Called by cmnn\_run.run.<br>
Plots the true vs. the photometric redshifts as a 2d histogram. User-specified options to represent outliers with colored points, and/or to draw polygons.

**Outputs:** Saves plot to the output directory (tzpz.png).

### cmnn_analysis.make_stats_file
Called by cmnn\_run.run.<br>
Calculates the photo-z statistics in bins of photo-z. Statistical measures are based on the photo-$z$ error: $\Delta z_{1+z_p} = (z_t-z_p)/(1+z_p)$ where $z_t$ is the true redshift, and $z_p$ the photo-$z$, of the test-set galaxy. For some statistics, catastrophic outlier rejection (COR) is done first, and test galaxies in the bin with $|z_t-z_p| >$ stats\_COR (default 1.5) are rejected. For a full description of the statistical measures, see Section 2 of Graham et al. (2020).

**Outputs:** Writes statistics to file in the output directory (stats.dat).

| Column Name | Description |
| :-- | :-- |
| meanz    | the mean zphot of galaxies in bin |
| CORmeanz | post-COR mean zphot of galaxies in bin |
| fout    | fraction of outliers (see note below) |
| stdd    | standard deviation in $\Delta z_{1+z_p}$ of all galaxies in bin |
| bias    | mean $\Delta z_{1+z_p}$ of all galaxies in bin |
| IQR     | interquartile range of $\Delta z_{1+z_p}$ |
| IQRstdd | stdandard deviation from the IQR ( = IQR / 1.349 ) |
| IQRbias | bias of test galaxies in the IQR  |
| CORstdd     | post-COR stdd |
| CORbias     | post-COR bias |
| CORIQR      | post-COR IQR |
| CORIQRstdd  | post-COR IQR stdd |
| CORIQRbias  | post-COR IQR bias |
| estdd       | error in stdd |
| ebias       | error in bias |
| eIQR        | error in IQR |
| eIQRstdd    | error in IQR stdd |
| eIQRbias    | error in IQR bias |
| eCORstdd    | error in post-COR stdd |
| eCORbias    | error in post-COR bias |
| eCORIQR     | error in post-COR IQR |
| eCORIQRstdd | error in post-COR IQR stdd |
| eCORIQRbias | error in post-COR IQR bias |


### cmnn_analysis.make_stats_plots
Called by cmnn\_run.run.<br>
Plots the photo-z statistics as a function of photo-z bin for a _runid_ (from stats.dat). By default, three plots for three statistics are made: robust standard deviation, robust bias, and fraction of outliers. The default is to apply catastrophic outlier rejection (COR) to the robust standard deviation and robust bias statistics (see user input stats\_COR for cmnn\_run.run).

**Outputs:** Saves plots to the output directory (CORIQRstdd.png, CORIQRbias.png, fout.png).

**Options:** When run stand-alone (i.e., not from cmnn\_run.run), the user may specify which of the statistical measures of photo-z quality a plot should be created for, and/or whether multiple _runids_ be co-plotted. When multiple _runids_ are co-plotted, the plots are saved to the directory "output/stats_plots/" with names formatted like "fout\_runid1\_runid2\_runid3.png".

| Inputs | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| user\_stats | str[M] | ['fout','CORIQRstdd','CORIQRbias'] | list of statistics for which to make plots |
| show\_SRD | bool | True | if True, the SRD target values are shown as dashed horizontal lines (ls.st/lpm-17) |
| multi\_run\_ids | str[N] | [runid] | array of multiple run ids to co-plot |
| multi\_run\_labels | str[N] | ['run '+runid] | array of legend labels that describe each run |
| multi\_run\_colors | str[N] | ['blue','orange','red','green','darkviolet'] | array of color names to use for each run (OK to pass $>5$ if $N>5$) |

### cmnn_analysis.get_stats
Called by cmnn\_analysis.make\_stats\_file.<br>
For a full set of true and photometric redshifts, returns the statistics for a given redshift bin. 

### cmnn_tools.convert_visits_to_depths
The user may specify the number of standard 30 second visits to be done in each filter _ugrizy_.<br>
Returns the 5$\sigma$ limiting magnitude depths in _ugrizy_ as a 6-element array.

```
>>> import cmnn_tools
>>> results_A = cmnn_tools.convert_visits_to_depths( [28, 40, 92, 92, 80, 80] )
>>> print(results_A)
[ 25.70894754  27.00257499  27.15473478  26.45473478  25.67886248 24.47886248]
```

### cmnn_tools.convert_year_to_depths_baseline
The user may specify the number of years of the baseline LSST observing strategy.<br>
Returns the $5{\sigma}$ limiting magnitudes depths in _ugrizy_ as a 6-element array.

```
>>> import cmnn_tools
>>> results_B = cmnn_tools.convert_year_to_depths_baseline( 5 )
>>> print(results_B)
[ 25.70894754  27.00257499  27.15473478  26.45473478  25.67886248  24.47886248]
```


## References
Graham et al. 2018, AJ, 155, 1 <br>
Graham et al. 2020, AJ, arXiv:2004.07885 <br>
Ivezić et al. 2019, ApJ, 873, 2
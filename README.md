# level2_analysis

Collection of function to deal with the level 2 data from the GROMORA project. See the main repository of the project at: [GROMORA-harmo](https://git.iap.unibe.ch/IAP_MCH/Level1_Analysis).

This is the main repository for all the data analysis done with the level 2 from GROMOS and SOMORA. It contains some diagnostics module used for the time series documentation [level2_gromora_diagnostics](level2_gromora_diagnostics.py) and the main comparison modules between GROMOS and SOMORA ([compare_gromora_v2](compare_gromora_v2.py)).

It also contains function for the comparison of these data with different datasets:
* [MLS](MLS.py)
* [SBUV/2](sbuv.py)
* [ECMWF](ecmwf.py)
* [MERRA2](merra2.py)

Also note the [plot_L2_paper](plot_L2_paper.py) which can be used to reproduce all plots from the GROMORA manuscript.
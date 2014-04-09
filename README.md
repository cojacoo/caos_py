caos_efu
========

An eco-hydrological model for observation and experiment driven water, solutes and energy dynamics simulation based on a representative macropore-matrix setup, explicit macropore-matrix interaction and water as particles. This model developed is part of the DFG research group Catchments As Organized Systems (CAOS).

we are still developing - pleas check back soon.
------------------------------------------------

Dependencies
------------

The following packages are required by the model, which on most systems can conveniently be installed via pip
* numpy
* scipy
* matplotlib
* pandas
* scikit-image
* shapely
* descartes

Although not required, the software is developed to run unsing ipython notebook. Notebooks for testing are included. You may need to have cython, lapack, geos and pil running to use these packages. Please refer to the respective repositories for their dependencies and installation procedures.

The software is given under the science commons license and still under development. It is explicitly given without any warranty. It may cause any imaginable and unintended damage to your system and connected parts. Do not use the software for other purpose but testing and experimenting. It is free to use, share and fork with explicitly naming the authors as doi:xxxxx. The testing data set is given under the same conditions.


Quickstart
----------

For a throughout description of the model concept please refer to doi:xxxxx. A full documentation will be given asap. Until then feel free to contact me for requests and assistance.

##Parameterization
The main model parameters are set in mcini.py. Here some files for experimental and observational records need to be set.

precf='irr_testcase.dat'    #top_boundary_flux_precip

etf='etp.dat'               #top_boundary_flux_etp

inimf='moist_testcase.dat'  #initial_moisture
Give the initial conditions either as table of theta or psi as no., starting depth, end depth, theta value, psi value.

macbf='macbase.dat'         #macropore definition file
There are two ways to parameterize macropores. One is via a stack of rectified horizontal images of Brilliant Blue dye tracer sprinkling experiments. The list of files needs to given as table holding depth, path to file and threshold for automated patch recognition.

tracerbf='tracer.dat'       #tracer base file
This file holds the table of recovered tracer concentrations in a vertical profile of several columns. The grid definitions and experiment conditions are set through the other tracer_* parameters accordingly. This information is used for calculation of the asymptotic advective flow field.

matrixbf='matrix.dat'       #matrix base file
This table of ID, saturated hydraulic conductivity [m/s], saturated soil moisture [cm3/cm3], residual soil moisture [cm3/cm3], van Genuchten alpha, van Genuchten n, bulk density rho and depth of sampling z [m] shall hold all soil matrix references of the project.

matrixdeffi='matrixdef.dat' #matrix definition file
This file holds the vertical profile of given matrix definitions in the format ID, start depth, end depth, trust. The idea behind is, that matrix definitions are gathered from analysis of multiple core samples at different depths and that the model shall handle any additional information given. However, you may accept one sample as more representative over others. The trust factor allows that reasonable but more exceptional samples may still be included but are used for the soil column setup with a lower probability.


##Testing
If you have iPython, simply change in the directory of the model setup and open a new notebook:

    ipython notebook --pylab=inline

In the dashboard you will find several testing setups which you can run.

Otherwise you may run the provided test*.py scripts.




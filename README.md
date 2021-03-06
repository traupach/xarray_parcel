# xarray-enabled atmospheric air parcel calculations

This repository contains code for [xarray](http://xarray.pydata.org/en/stable/)-enabled versions of air parcel functions, to calculate convective available potential energy (CAPE), convective inhibition (CIN), levels of condensation and free convection, parcel profiles, etc, in vectorised form. The functions are based on [MetPy](https://unidata.github.io/MetPy/latest/index.html) functions and are thus released under the same [license](LICENSE).

The main module file is `modules/parcel_functions.py`. Testing functions are in `modules/parcel_test.py` and copies of MetPy unit tests are in `modules/unit_tests.py`. `test_data.nc` is testing data from subset from the [Aus400](http://climate-cms.wikis.unsw.edu.au/Aus400) simulations (released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)). (Note: the functions in `modules/moist_lapse_analytic.py` are an attempt to make a vectorised version of `moist_lapse`, but are slow and inaccurate and should therefore not be used).

A [demo notebook](parcel_functions_demo.ipynb) shows testing results and benchmarking results, and explains how moist adiabats are approximated using a lookup table.

**Important note:** the vertical dimension of input data must be specified must relate to the model level number from the surface to the top of the model (meaning the first element along the vertical dimension is the lowest atmospheric level). Use the function `valid_data` to check that data is valid for use with this package.

**As stated in the [license](LICENSE) this software is provided as-is without any guarantee of accuracy or fit for purpose.**

This implementation was coded by Tim Raupach, a postdoc at the UNSW Sydney [Climate Change Research Centre](https://ccrc.unsw.edu.au).

## Notes:

- It is important to note that moist adiabats are **approximated** in this implementation by using a lookup table for speed. See the [demo notebook](parcel_functions_demo.ipynb) for details.
- In `add_lcl_to_profile`, a linear interpolation for temperature on pressure coordinates is used, in order to match MetPy's implementation. In future it may be more accurate to replace the linear interpolation with an interpolation on log-transformed coordinates.

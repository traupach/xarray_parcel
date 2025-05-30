# parcel_functions.py
#
# xarray-enabled versions of MetPy functions for atmospheric parcel
# calculations.
# 
# Author: Tim Raupach <t.raupach@unsw.edu.au>

import sys
import time
import metpy
import xarray
import metpy.calc
import numpy as np
from metpy.units import units
import metpy.constants as mpconsts
from numba import float64, guvectorize

# Class global variables to contain moist adiabat lookup tables.
this = sys.modules[__name__]
this.moist_adiabat_lookup = None
this.moist_adiabats = None

@guvectorize(
    "(float64[:], float64[:], float64[:], float64[:])",
    "(m), (n), (n) -> (m)",
    nopython=True,
)
def interp1d_numba(at, xp, fp, out):
    # Perform simple 1d interpolation with numpy and no data checking.
    #
    # Arguments:
    #    at: coordinates at which to find interpolated values.
    #    xp: known coordinates. Must be monotonically increasing.
    #    fp: values at known coordinates.
    #
    # Returns: interpolated values of fp at 'at' points.
    out[:] = np.interp(at, xp, fp)

def load_moist_adiabat_lookups(**kwargs):
    """
    Load cache adiabat lookup tables ready for use in other functions.
    If cache doesn't yet exist, use moist_adiabat_tables to generate 
    them.
    
    Arguments:
    
        - Chunks: Chunk definition to use. Default for 28 processors.
        - **kwargs: Any arguments to moist_adiabat_tables().
    """
    
    this.moist_adiabat_lookup, this.moist_adiabats = moist_adiabat_tables(
        regenerate=False, cache=True, **kwargs)
    this.moist_adiabat_lookup = this.moist_adiabat_lookup.load()
    this.moist_adiabats = this.moist_adiabats.sortby('pressure').load()
        
def lookup_tables_loaded():
    """
    Ensure that lookup tables are loaded and throw an error if not.
    """
    assert this.moist_adiabat_lookup is not None, 'Call load_moist_adiabat_lookups first.'
    assert this.moist_adiabats is not None, 'Call load_moist_adiabat_lookups first.'
        
def get_layer(dat, depth=100, vert_dim='model_level_number', interpolate=True):
    """
    Return an atmospheric layer from the surface with a given depth.

    Arguments:

      - dat: DataArray, must contain pressure.
      - depth: Depth above the bottom of the layer to mix [hPa].
      - vert_dim: Vertical dimension name.
      - interpolate: Interpolate the bottom/top layers?

    Returns:

      - xarray DataArray with pressure and data variables for the layer.
    """
       
    # Use the surface (lowest level) pressure as the bottom pressure.
    bottom_pressure = dat.pressure.max(dim=vert_dim)
        
    # Calculate top pressure.
    if interpolate:
        top_pressure = bottom_pressure-depth
        interp_level = log_interp(x=dat, at=top_pressure, 
                                  coords=dat.pressure, dim=vert_dim)
        interp_level['pressure'] = top_pressure
        
        dat = insert_level(d=dat, level=interp_level, coords='pressure',
                           vert_dim=vert_dim)
    else:
        top_pressure = bound_pressure(pressure=dat.pressure, 
                                      bound=bottom_pressure-depth, 
                                      vert_dim=vert_dim)
        
    # Select the layer.
    layer = dat.where(dat.pressure <= bottom_pressure)
    layer = layer.where(layer.pressure >= top_pressure)
    
    return layer

def most_unstable_parcel(dat, depth=300, vert_dim='model_level_number'):
    """
    Return the most unstable parcel with an atmospheric layer from
    with the requested bottom and depth. No interpolation is
    performed. If there are multiple 'most unstable' parcels, return
    the first in the vertical array.

    Arguments:

        - dat: DataArray, must contain pressure, temperature, and dewpoint.
        - bottom: Pressure level to start from [hPa].
        - depth: Depth above the bottom of the layer to mix [hPa].
        - drop: Drop unselected elements?
        - vert_dim: Vertical dimension name.

    Returns:

        - xarray DataArray with pressure and data variables for the layer.
    """

    layer = get_layer(dat=dat, depth=depth, vert_dim=vert_dim, interpolate=False)
    eq = metpy.calc.equivalent_potential_temperature(
        pressure=layer.pressure,
        temperature=layer.temperature,
        dewpoint=layer.dewpoint).metpy.dequantify()
    max_eq = eq.max(dim=vert_dim)
    pres = layer.where(eq == max_eq).pressure.max(dim=vert_dim)
    
    counts = layer.pressure.where(layer.pressure == pres).count(dim=vert_dim).where(~np.isnan(pres))
    assert counts.max() == counts.min() == 1, 'Vertical pressures are not unique'
    
    most_unstable = layer.where(layer.pressure == pres).max(dim=vert_dim,
                                                            keep_attrs=True)
    return most_unstable
    
def mixed_layer(dat, depth=100, vert_dim='model_level_number'):
    """
    Mix variable(s) over a layer, yielding a mass-weighted average.

    Integrate a data variable with respect to pressure and determine the
    average value using the mean value theorem.

    Arguments:

        - dat: The DataArray to mix. Must contain pressure and variables.
        - bottom: Pressure above the surface pressure to start from [hPa].
        - depth: Depth above the bottom of the layer to mix [hPa].
        - vert_dim: The name of the vertical dimension.

    Returns:

        - xarray with mixed values of each data variable.
    """
    
    layer = get_layer(dat=dat, depth=depth, vert_dim=vert_dim)
    
    pressure_depth = np.abs(layer.pressure.min(vert_dim) - 
                            layer.pressure.max(vert_dim))
   
    ret = (1. / pressure_depth) * trapz(dat=layer, x='pressure', dim=vert_dim)
    return ret
    
def trapz(dat, x, dim, mask=None, only_positive=False, only_negative=False):
    """ 
    Perform trapezoidal rule integration along an axis, ala numpy.trapz.
    Estimates int y dx.
   
    Arguments:

        - dat: Data to process.
        - x: The variable that contains 'x' values along dimension 'dim'.
        - dim: The dimension along which to integrate 'y' values.
        - mask: A mask the size of dx/means (ie dim.size-1) for which 
                areas to include in the integration.
        - only_positive, only_negative: Include only positive or negative values of the
                                        area?

    Returns:

        - Integrated value along the axis.
    """

    assert np.all(np.abs(dat[dim].diff(dim=dim)) == 1), 'Index increments must all be 1.'
    
    dx = np.abs(dat[x].diff(dim))
    dx = dx.reset_coords(drop=True)
    means = dat.rolling({dim: 2}, center=True).mean(keep_attrs=True)
    means = means.reset_coords(drop=True)

    dx = dx.assign_coords({dim: dx[dim]-1})
    means = means.assign_coords({dim: means[dim]-1})
    
    if mask is not None:
        dx = dx.where(mask)
        means = means.where(mask)
    
    areas = dx * means
    
    assert not (only_positive and only_negative), 'Only negative OR positive regions can be included in trapz.'
    if only_positive:
        areas = areas.where(areas > 0)
    if only_negative:
        areas = areas.where(areas < 0)
    
    return areas.sum(dim)
    
def bound_pressure(pressure, bound, vert_dim='model_level_number'):
    """
    Calculate the bounding pressure in a layer; returns the closest
    pressure to the bound.  If two pressures are equally distant from
    the bound, the larger pressure is returned.
    
    Arguments:

        - pressure: Atmospheric pressures [hPa].
        - bound: Bound to retrieve, broadcastable to pressure [hPa].

    Returns:

        - The bound pressures.
    """
    
    diffs = np.abs(pressure - bound)
    bounds = pressure.where(diffs == diffs.min(dim=vert_dim))
    bounds = bounds.max(dim=vert_dim).squeeze(drop=True)
    return bounds

def mixed_parcel(pressure, temperature, dewpoint, depth=100,
                 vert_dim='model_level_number'):
    """
    Fully mix a layer of given depth above the surface and find the temparature,
    pressure and dewpoint of the parcel.

    Arguments:

        - pressure: Pressure by level [hPa].
        - temperature: Temperature at each level [K].
        - dewpoint: Dewpoint at each level [K].
        - depth: Depth above the surface to mix [hPa].
        - vert_dim: The name of the vertical dimension.

    Returns:

        - DataArray with mixed parcel pressure [hPa], temperature [K]
          and dewpoint [K].
    """
    
    # Use the surface (lowest level) pressure as the start of the layer to mix.
    parcel_start_pressure = pressure.isel({vert_dim: 0})

    # Calculate the potential temperature over the layer.
    theta = metpy.calc.potential_temperature(pressure, temperature)
    theta = theta.metpy.dequantify()
    theta.name = 'theta'
    
    # Mixing ratio over the layer.
    mixing_ratio = metpy.calc.saturation_mixing_ratio(pressure, dewpoint)
    mixing_ratio = mixing_ratio.metpy.dequantify()
    mixing_ratio.name = 'mixing_ratio'
    
    # Mix theta and mixing ratio over the layer.
    assert pressure.name is not None, 'pressure requires name pressure.'
    mp = mixed_layer(xarray.merge([pressure, theta, mixing_ratio]), depth=depth,
                     vert_dim=vert_dim)
        
    # Convert potential temperature back to temperature.
    mp['temperature'] = (mp.theta *
                         metpy.calc.exner_function(parcel_start_pressure))
    mp['temperature'] = mp.temperature.metpy.dequantify()
    mp.temperature.attrs['long_name'] = 'Mixed parcel temperature'
    mp.temperature.attrs['units'] = 'K'

    # Convert mixing ratio back to dewpoint.
    mp['vapour_pressure'] = metpy.calc.vapor_pressure(parcel_start_pressure,
                                                      mp.mixing_ratio)
    mp['vapour_pressure'] = mp.vapour_pressure.metpy.dequantify()
    mp.vapour_pressure.attrs['long_name'] = 'Mixed-parcel vapour pressure'
    
    mp['dewpoint'] = metpy.calc.dewpoint(mp.vapour_pressure)
    mp['dewpoint'] = mp.dewpoint.metpy.convert_units('K')
    mp['dewpoint'] = mp.dewpoint.metpy.dequantify()
    mp.dewpoint.attrs['long_name'] = 'Mixed-parcel dewpoint'
    
    # For pressure, use the starting pressure for the layer (following MetPy's 
    # mixed_parcel function).
    mp['pressure'] = parcel_start_pressure 
    
    return mp

def dry_lapse(pressure, parcel_temperature, parcel_pressure=None,
              vert_dim='model_level_number'):
    """
    Calculate the temperature of a parcel raised dry-adiabatically (conserving
    potential temperature).

    Arguments:

        - pressure: Atmospheric pressure level(s) of interest [hPa].
        - parcel_temperature: Parcel temperature before lifting
          (constant or broadcast-able DataArray).
        - parcel_pressure: Parcel pressure(s) before lifting. Defaults
          to vertical maximum.
        - vert_dim: The name of the vertical dimension.

    Returns:

        - Parcel temperature at each pressure level.
    """
    
    if parcel_pressure is None:
        parcel_pressure = pressure.max(vert_dim)
    out = parcel_temperature * (pressure / parcel_pressure)**mpconsts.kappa
    out.attrs['long_name'] = 'Dry lapse rate temperature'
    out.attrs['units'] = 'K'
    return out

def moist_adiabat_tables(regenerate=False, cache=True, chunks=None, base_dir='.',
                         lookup_cache='/adiabat_lookups/moist_adiabat_lookup.nc',
                         adiabats_cache='/adiabat_lookups/adiabats_cache.nc',
                         **kwargs):
    """
    Calculate moist adiabat lookup tables.
    
    Arguments:

        - regenerate: Calculate from scratch and save caches?
        - cache: Write cache files?
        - chunks: Chunks argument for xarray .chunk() function.
        - base_dir: The base directory in which to read/write caches.
        - lookup_cache: A cache file (nc) for the adiabat lookup table.
        - adiabats_cache: A cache file (nc) for the adiabats cache.
        - **kwargs: Keyword arguments to moist_adiabat_lookup().
                           
    Returns:

        - two DataArrays: 1) a lookup table of pressure/temperature
          vs. adiabat number, and 2) a lookup table of adiabat number
          to temperature by pressure profiles.
    """
    
    if not regenerate:
        adiabat_lookup = xarray.open_dataset(base_dir + lookup_cache, 
                                             chunks=chunks).persist()
        adiabats = xarray.open_dataset(base_dir + adiabats_cache, 
                                       chunks=chunks).persist()
        return adiabat_lookup, adiabats
    
    # Generate lookup tables.
    adiabat_lookup, adiabats = moist_adiabat_lookup(**kwargs)
    
    if cache:
        adiabats.to_netcdf(base_dir + adiabats_cache)
        adiabat_lookup.to_netcdf(base_dir + lookup_cache)
        
    return adiabat_lookup.chunk(chunks), adiabats.chunk(chunks)

def round_to(x, to, dp=2):
    """
    Round x to the nearest 'to' and return rounded to 'dp' decimal points.
    """
    return np.round(np.round(x / to) * to, dp)

def wet_bulb_temperature_fast(temperature, dewpoint):
    """
    Use a fast method to estimate wet bulb temperature. The method is the "1/3" rule 
    discussed in Knox et al., 2017 (https://doi.org/10.1175/BAMS-D-16-0246.1). Its 
    error is shown in their Figure 3.
    
    Arguments:
    
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint temperatures [K].
        
    Returns:
        
        - Estimated wet-bulb temperature.
    """
    
    wb = temperature - (1/3)*(temperature-dewpoint)
    
    wb.name = 'wet_bulb_temperature'
    wb.attrs['long_name'] = 'Wet bulb temperature'
    wb.attrs['description'] = 'Estimated using 1/3 method.'
    wb.attrs['units'] = 'K'
        
    return wb

def wet_bulb_temperature(pressure, temperature, dewpoint, vert_dim='model_level_number'):
    """
    Calculate wet-bulb pressure using "Normand's rule" -- see Knox et al., 2017 
    (https://doi.org/10.1175/BAMS-D-16-0246.1) for a description of the method.

    Arguments:
        - pressure: Pressure level(s) of interest [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint temperatures [K].
        - vert_dim: The vertical dimension to operate on.
        
    Returns:
        - The wet bulb temperature of each point.
    """
    
    if pressure.chunks is not None:
        print('WARNING: wet_bulb_temperatuer function uses a for loop that ' +
              'performs badly when dask is used. Loading data into memory. ' +
              'If an approximation will do, use wet_bulb_temperature_fast.')
        pressure = pressure.load()
        temperature = temperature.load()
        dewpoint = dewpoint.load()
    
    # For each point, lift up the dry adiabat until we reach the LCL, then
    # bring the lifted parcel down the moist adiabat to the original pressure
    # to find the wet-bulb temperature.
    
    if vert_dim in pressure.coords:
        ml = xarray.zeros_like(pressure)
        for v in pressure[vert_dim]:
            
            lcls = lcl(parcel_pressure=pressure.sel({vert_dim: v}), 
                       parcel_temperature=temperature.sel({vert_dim: v}), 
                       parcel_dewpoint=dewpoint.sel({vert_dim: v}))

            p = pressure.sel({vert_dim: v}).expand_dims(vert_dim)
            ml.loc[{vert_dim: v}] = moist_lapse(pressure=p,
                                                parcel_temperature=lcls.lcl_temperature,
                                                parcel_pressure=lcls.lcl_pressure,
                                                vert_dim=vert_dim).compute().squeeze().reset_coords(drop=True)
            del lcls

    else:
        lcls = lcl(parcel_pressure=pressure, parcel_temperature=temperature, 
                   parcel_dewpoint=dewpoint)

        ml = moist_lapse(pressure=pressure,
                         parcel_temperature=lcls.lcl_temperature,
                         parcel_pressure=lcls.lcl_pressure,
                         vert_dim=vert_dim)

    ml = ml.reset_coords(drop=True)
    ml.name = 'wet_bulb_temperature'
    ml.attrs['long_name'] = 'Wet bulb temperature'
    ml.attrs['units'] = 'K'

    return(ml)

def moist_adiabat_lookup(pressure_levels=np.round(np.arange(1100, 2,
                                                            step=-0.5), 1),
                         temperatures=np.round(np.arange(173, 316,
                                                         step=0.02), 2),
                         pres_step=0.5, temp_step=0.02):
    """
    Calculate moist adiabat lookup tables.
    
    Arguments:
        - pressure_levels: Pressure levels to keep in adiabat lookup
                           table [hPa].
        - temperatures: Temperatures to keep in adiabat lookup table [K].
        - pres_step, temp_step: (Positive) step size for
                                pressure_levels and temperatures,
                                respectively.
                              
    Returns:
    
        - two DataArrays: 1) a lookup table of pressure/temperature
          vs. adiabat number, and 2) a lookup table of adiabat number
          to temperature by pressure profiles.
    """
        
    curves = []
    adiabat_lookup = xarray.Dataset({'adiabat': np.nan})
    adiabat_lookup = adiabat_lookup.expand_dims({'pressure': pressure_levels, 
                                                 'temperature':
                                                 temperatures}).copy(deep=True)
    
    # Find the adiabat for each starting temperature.
    i = 1
    for parcel_temperature in temperatures:
        for offset in [0, temp_step/2]:
            profile_temps = metpy.calc.moist_lapse(
                pressure=pressure_levels*units.hPa,
                temperature=(parcel_temperature+offset)*units.K).m

            nearest_temps = round_to(profile_temps, temp_step)
            idx = np.isin(nearest_temps, temperatures)
            temp_idx = xarray.DataArray(nearest_temps[idx], dims=['idx'])
            pres_idx = xarray.DataArray(pressure_levels[idx], dims=['idx'])
            adiabat_lookup.adiabat.loc[{'pressure':pres_idx,
                                        'temperature': temp_idx}] = i

            # In profile_temps we have the adiabat temperature for
            # every pressure level.  But some temperatures in the
            # lookup table may be missing. Interpolate the pressures
            # for each required temperature.
            pres_per_temp = np.interp(x=temperatures, xp=profile_temps[::-1], 
                                      fp=pressure_levels[::-1], right=np.nan,
                                      left=np.nan)

            pres_per_temp = round_to(pres_per_temp, pres_step)
            idx = np.isin(pres_per_temp, pressure_levels)
            pres_idx = xarray.DataArray(pres_per_temp[idx], dims=['idx'])
            temp_idx = xarray.DataArray(temperatures[idx], dims=['idx'])
            adiabat_lookup.adiabat.loc[{'pressure':pres_idx,
                                        'temperature': temp_idx}] = i

            # curves contains the adiabats themselves.
            curves.append(xarray.Dataset({'temperature': (['pressure'],
                                                          profile_temps)},
                                         coords={'pressure': pressure_levels}))
            curves[-1]['adiabat'] = i
            i = i + 1

    # Combine curves into one dataset.
    adiabats = xarray.combine_nested(curves, concat_dim='adiabat')
        
    for x in [adiabat_lookup, adiabats]:
        x.pressure.attrs['long_name'] = 'Pressure'
        x.pressure.attrs['units'] = 'hPa'
        x.temperature.attrs['long_name'] = 'Temperature'
        x.temperature.attrs['units'] = 'K'
        x.adiabat.attrs['long_name'] = 'Adiabat index'
        
    return adiabat_lookup, adiabats

def moist_lapse(pressure, parcel_temperature, parcel_pressure=None,
                vert_dim='model_level_number', persist=True):
    """
    Return the temperature of parcels raised moist-adiabatically
    (assuming liquid saturation processes).  Note: What is returned
    are approximate pseudo-adiabatic moist lapse rates, found using a
    lookup table.

    Arguments:

        - pressure: Atmospheric pressure(s) to lift the parcel to [hPa].
        - parcel_temperature: Temperature(s) of parcels to lift [K].
        - parcel_pressure: Parcel pressure before lifting. Defaults to lowest 
                           vertical level.
        - vert_dim: The name of the vertical dimension.
        - persist: Persist adiabat lookups to memory for speed improvements at cost of RAM?
        
    Returns:

        - Temperature of each parcel lifted to each pressure level.
    """

    lookup_tables_loaded()
    
    if parcel_pressure is None:
        parcel_pressure = pressure.isel({vert_dim: 0})
        
    # For each starting parcel, find the moist adiabat that intersects
    # the parcel pressure and temperature.
    adiabat_idx = this.moist_adiabat_lookup.sel({'pressure': parcel_pressure,
                                                 'temperature': parcel_temperature},
                                                 method='nearest')
    adiabat_idx = adiabat_idx.adiabat.reset_coords(drop=True)
    
    # Chunks from DataArrays are not named; whereas from DataSets they are. So 
    # convert to DataSet just to get chunk information.
    chunks = xarray.Dataset({'pressure': pressure}).chunks
    if isinstance(pressure, xarray.DataArray) and pressure.chunks is not None:
        adiabat_idx = adiabat_idx.chunk({x: chunks[x] for x in adiabat_idx.dims})
        pressure = pressure.chunk({vert_dim: -1})
        if persist:
            adiabat_idx = adiabat_idx.persist()
        
    # Replace points without an adiabat with an index (1) so the lookup works;
    # these will be set to nans later.
    valid = np.logical_not(np.isnan(adiabat_idx))
    adiabat_idx = adiabat_idx.where(valid, other=1)
    adiabats = this.moist_adiabats.sel(adiabat=adiabat_idx)
    adiabats = adiabats.squeeze().reset_coords(drop=True)
    
    if isinstance(pressure, xarray.DataArray) and pressure.chunks is not None:
        adiabats = adiabats.chunk({x: chunks[x] for x in [y for y in adiabats.dims if y in chunks]})
        adiabats = adiabats.chunk({'pressure': -1})
        if persist:
            adiabats = adiabats.persist()
        
    # Reset replaced points.
    adiabats = adiabats.where(valid, other=np.nan)
    
    assert np.all(adiabats.pressure.diff('pressure') >= 0), 'Adiabats must be sorted by increasing pressure.'
    out = xarray.apply_ufunc(
        interp1d_numba,
        pressure,              # Interpolate for pressures at each level.
        adiabats.pressure,     # Adiabat pressures to interpolate.
        adiabats.temperature,  # Adiabat temperatures to interpolate.
        input_core_dims=[[vert_dim], ['pressure'], ['pressure']],
        output_core_dims=[[vert_dim]],
        dask='parallelized')
        
    # Add metadata.
    out.attrs['long_name'] = 'Moist lapse rate temperature'
    out.attrs['units'] = 'K'
    
    # Don't allow extrapolation.
    out = out.where(pressure >= adiabats.pressure.min().values)
    out = out.where(pressure <= adiabats.pressure.max().values)
    
    # Don't return values where inputs are nan.
    out = out.where(~np.isnan(parcel_temperature))
    out = out.where(~np.isnan(parcel_pressure))
    out = out.where(~np.isnan(pressure))
    
    return out

def lcl(parcel_pressure, parcel_temperature, parcel_dewpoint):
    """
    Return the lifting condensation level for parcels.
    
    Arguments:

        - parcel_pressure: Pressure of the parcel to lift [hPa].
        - parcel_temperature: Parcel temperature [K].
        - parcel_dewpoint: Parcel dewpoint [K].
    
    Returns:

        - A Dataset with lcl_pressure and lcl_temperature.
    """
    
    # MetPy LCL can't handle nans. Replace NaNs with a valid set of 
    # pressure/temperature/dewpoint for lcl to use; then discard results
    # before returning.
    valid_points = np.logical_not(np.logical_or(np.logical_or(np.isnan(parcel_pressure),
                                                              np.isnan(parcel_temperature)),
                                                np.isnan(parcel_dewpoint)))
    valid_points = valid_points.reset_coords(drop=True)
    
    parcel_pressure = parcel_pressure.where(valid_points, other=1000)
    parcel_temperature = parcel_temperature.where(valid_points, other=273.15)
    parcel_dewpoint = parcel_dewpoint.where(valid_points, other=273.15)
    
    parcel_pressure.name = 'parcel_pressure'
    parcel_temperature.name = 'parcel_temperature'
    parcel_dewpoint.name = 'parcel_dewpoint'
    obj = xarray.merge([parcel_pressure, parcel_temperature, parcel_dewpoint])
    obj = obj.reset_coords(drop=True)

    # Define a block-able function for metpy's LCL. 
    def lcl_block(obj):
        press_lcl, temp_lcl = metpy.calc.lcl(pressure=obj.parcel_pressure,
                                             temperature=obj.parcel_temperature,
                                             dewpoint=obj.parcel_dewpoint)

        assert np.all(obj.parcel_pressure.dims == obj.parcel_temperature.dims)
        assert np.all(obj.parcel_pressure.dims == obj.parcel_dewpoint.dims)
        dims = obj.parcel_pressure.dims
        
        # Calculate virtual temperature at LCL (at LCL, temperature == dewpoint).
        lcl_mixing_ratio = mixing_ratio(temperature=temp_lcl, 
                                        dewpoint=temp_lcl,
                                        pressure=press_lcl)
        lcl_virt_temp = virtual_temperature(temperature=temp_lcl,
                                            mixing_ratio=lcl_mixing_ratio)

        res =  xarray.Dataset({'lcl_pressure': (dims, press_lcl.m),
                               'lcl_temperature': (dims, temp_lcl.m),
                               'lcl_virtual_temperature': (dims, lcl_virt_temp.m)},
                               coords={x: obj.coords[x].values for x in dims})

        return(res)

    # Apply the LCL function one block (chunk) at a time, in parallel.
    out = obj.map_blocks(lcl_block)

    out.lcl_pressure.attrs['long_name'] = ('Lifting condensation ' +
                                           'level pressure')
    out.lcl_pressure.attrs['units'] = 'hPa'
    out.lcl_temperature.attrs['long_name'] = ('Lifting condensation ' +
                                              'level temperature')
    out.lcl_temperature.attrs['units'] = 'K'
    out.lcl_virtual_temperature.attrs['long_name'] = ('Lifting condensation ' +
                                                      'level virtual temperature')
    out.lcl_virtual_temperature.attrs['units'] = 'K'
    
    # Only return valid points.
    out = out.where(valid_points)
    
    return out

def mixing_ratio(temperature, dewpoint, pressure):
    """
    Calculate mixing ratio.
    
    Arguments:
        - temperature: Temperature [K].
        - dewpoint: Dewpoint [K].
        - pressure: Pressure [hPa].
    
    Returns:
        
        - Mixing ratio [kg kg-1].
    """

    relative_humidity = metpy.calc.relative_humidity_from_dewpoint(
        temperature=temperature,
        dewpoint=dewpoint)
    res = metpy.calc.mixing_ratio_from_relative_humidity(
        pressure=pressure,
        temperature=temperature,
        relative_humidity=relative_humidity)
    
    if isinstance(res, xarray.DataArray):
        res = res.metpy.dequantify()
        res.attrs['units'] = 'kg kg$^{-1}$'
        
    return res

def parcel_profile(pressure, parcel_pressure, parcel_temperature, parcel_dewpoint, 
                   vert_dim='model_level_number'):
    """
    Calculate temperatures of a lifted parcel.
    
    Arguments:

        - pressure: Pressure levels to calculate on [hPa].
        - parcel_pressure: Pressure of the parcel [hPa].
        - parcel_temperature: Temperature of the padrcel [K].
        - parcel_dewpoint: Dewpoint of the parcel [K].
        - vert_dim: The name of the vertical dimension.
       
    Returns:

        - Dataset with the temperature of the parcel lifted from
          parcel_pressure to levels in pressures, plus the LCL
          pressure and temperature.
    """
       
    out = xarray.Dataset()
    out['pressure'] = pressure

    # Find the LCL for the selected parcel.
    out = xarray.merge([out, lcl(parcel_pressure=parcel_pressure, 
                                 parcel_temperature=parcel_temperature, 
                                 parcel_dewpoint=parcel_dewpoint)])

    # Parcels are raised along the dry adiabats from the starting
    # point to the LCL.
    below_lcl = dry_lapse(pressure=pressure, 
                          parcel_temperature=parcel_temperature, 
                          parcel_pressure=parcel_pressure,
                          vert_dim=vert_dim)
    
    # Along the dry adiabat the parcel's mixing ratio remains constant.
    parcel_mixing_ratio = mixing_ratio(temperature=parcel_temperature*units.K,
                                       dewpoint=parcel_dewpoint*units.K,
                                       pressure=parcel_pressure*units.hPa)
    
    # Above the LCL parcels follow the moist adiabats from the LCL
    # temp/pressure.
    above_lcl = moist_lapse(pressure=pressure, 
                            parcel_temperature=out.lcl_temperature,
                            parcel_pressure=out.lcl_pressure,
                            vert_dim=vert_dim)
    
    # Above the LCL, the mixing ratio is the saturation mixing ratio.
    mixing_ratios = metpy.calc.saturation_mixing_ratio(
        total_press=pressure, 
        temperature=above_lcl)
    mixing_ratios = mixing_ratios.metpy.dequantify()
    mixing_ratios.name = 'mixing_ratio'
    
    # Temperatures follow the dry/moist adiabat curves.
    out['temperature'] = below_lcl.where(pressure >= out.lcl_pressure,
                                         other=above_lcl)
    out.temperature.attrs['long_name'] = 'Lifted parcel temperature'
    out.temperature.attrs['units'] = 'K'
    
    # Apply the virtual temperature correction. 
    mixing_ratios = mixing_ratios.where(pressure <= out.lcl_pressure,
                                        other=parcel_mixing_ratio)
    out['virtual_temperature'] = virtual_temperature(
        temperature=out.temperature,
        mixing_ratio=mixing_ratios)
    
    out = out.reset_coords(drop=True)
    return out

def virtual_temperature(temperature, mixing_ratio, epsilon=0.608):
    """Calculate virtual temperature as per Doswell & Rasmussen 1994.
    
    Note, uses epsilon=0.608 from Doswell & Rasmussen 1994 by default.
    
    Arguments:
    
        - temperature: Air temperature [K].
        - mixing_ratio: Mixing ratio [g g-1].
        - epsilon: Value of epsilon constant to use.
        
    Returns:
    
        - The virtual temperature(s).
    """
    
    res = temperature * (1 + epsilon*mixing_ratio)
    
    if isinstance(res, xarray.DataArray):
        res.attrs['units'] = 'K'
        res.attrs['long_name'] = 'Virtual temperature'
        
    return res

def parcel_profile_with_lcl(pressure, temperature, dewpoint, parcel_pressure,
                            parcel_temperature, parcel_dewpoint,
                            vert_dim='model_level_number',
                            lcl_interp='log'):
    """
    Calculate temperatures of a lifted parcel, including at the lcl.
    
    Arguments:

        - pressure: Pressure levels to calculate on [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint at each pressure level [K].
        - parcel_pressure: Pressure of the parcel [hPa].
        - parcel_temperature: Temperature of the parcel [K].
        - parcel_dewpoint: Dewpoint of the parcel [K].
        - vert_dim: The name of the vertical dimension.
        - lcl_interp: Interpolator for lcl environment. 
  
    Returns:

         - Dataset with the temperature of the parcel lifted from
           parcel_pressure to levels in pressures, including the LCL,
           plus the LCL pressure and temperature, and environmental
           temperature including at the LCL.
    """
    
    profile = parcel_profile(pressure=pressure,
                             parcel_pressure=parcel_pressure,
                             parcel_temperature=parcel_temperature,
                             parcel_dewpoint=parcel_dewpoint,
                             vert_dim=vert_dim)
    
    # Calculate environmental virtual temperatures.
    mix_ratio = mixing_ratio(temperature=temperature, 
                             dewpoint=dewpoint,
                             pressure=pressure)
    virtual_temp = virtual_temperature(temperature=temperature,
                                       mixing_ratio=mix_ratio)
    
    environment = xarray.Dataset({'temperature': temperature,
                                  'virtual_temperature': virtual_temp,
                                  'dewpoint': dewpoint,
                                  'pressure': profile.pressure})
    environment.dewpoint.attrs['long_name'] = 'Environment dewpoint'
    environment.dewpoint.attrs['units'] = 'K'
    environment.temperature.attrs['long_name'] = 'Environment temperature'
    environment.temperature.attrs['units'] = 'K'
    
    return add_lcl_to_profile(profile=profile, vert_dim=vert_dim,
                              environment=environment, 
                              interpolator=lcl_interp)

def add_lcl_to_profile(profile, vert_dim='model_level_number',
                       environment=None, interpolator='log'):
    """
    Add the LCL to a profile.
    
    Arguments:

        - profile: Profile as returned from parcel_profile().
        - vert_dim: The vertical dimension to add the LCL pressure/temp to.
        - environment: The environment (e.g. temperature/virtual temp) 
                       to interpolate at the lcl_pressure.
        - interpolator: 'linear' or 'log' to use for vertical interpolation. 
        
    Returns:

        - A new profile object with LCL pressure and temperatures
          added. Note the vertical coordinate in the new profile is
          reindexed.
    """
    
    assert interpolator in ['linear', 'log'], 'interpolator must be linear or log'
    
    # The new level to add. 
    level = xarray.Dataset({'pressure': profile.lcl_pressure,
                            'temperature': profile.lcl_temperature,
                            'virtual_temperature': profile.lcl_virtual_temperature})
    out = insert_level(d=profile, level=level, coords='pressure',
                       vert_dim=vert_dim)
    out['lcl_pressure'] = profile.lcl_pressure
    out['lcl_temperature'] = profile.lcl_temperature
    out['lcl_virtual_temperature'] = profile.lcl_virtual_temperature
    out.temperature.attrs['long_name'] = 'Temperature at LCL'
    out.pressure.attrs['long_name']  = 'Pressure at LCL'
    out.lcl_virtual_temperature.attrs['long name'] = 'Virtual temperature at LCL'
    
    if not environment is None:
        # Interpolate the environment to get the level to insert. 
        # Note: MetPy uses a linear interpolator even on pressure levels.
        # By default I use the log interpolator for greater accuracy.
        if interpolator == 'linear':
            interp_level = linear_interp(x=environment,
                                         coords=environment.pressure,
                                         at=level.pressure,
                                         dim=vert_dim)
        elif interpolator == 'log':
            interp_level = log_interp(x=environment,
                                      coords=environment.pressure,
                                      at=level.pressure,
                                      dim=vert_dim)
        
        # Set the interpolated pressure.
        interp_level['pressure'] = level.pressure
        
        if 'virtual_temperature' in interp_level.keys():
            interp_level.dewpoint.attrs['units'] = 'K'
            interp_level.temperature.attrs['units'] = 'K'

            # Recalculate virtual temperature from interpolated dewpoint and temperature.
            mix_ratio = mixing_ratio(temperature=interp_level.temperature, 
                                     dewpoint=interp_level.dewpoint,
                                     pressure=interp_level.pressure)
            interp_level['virtual_temperature'] = virtual_temperature(temperature=interp_level.temperature,
                                                                      mixing_ratio=mix_ratio)
        
        # Add the new level into the environment.
        new_environment = insert_level(d=environment, level=interp_level, 
                                       coords='pressure', vert_dim=vert_dim)
        
        for k in environment.keys():
            if k != 'pressure':
                out['environment_'+k] = new_environment[k]
                out['environment_'+k].attrs = environment[k].attrs
        
    return out

def insert_level(d, level, coords, vert_dim='model_level_number',
                 fill_value=-999):
    """
    Insert a new level into a vertically sorted dataset.
    
    Arguments:

        - d: The data to work on.
        - level: The new values to add; a single layer with values for
                 'coord' and any other variables to add.
        - coords: The coordinate name in d.
        - vert_dim: The vertical dimension to add new level to.
        
    Returns:

        - A new object with the new level added.
    
    Note 1: the vertical coordinate in the new profile is reindexed.
    Note 2: if the coordinates ('coord') used to index the levels already exist 
            somewhere in 'd', the existing coordinate and its value is kept 
            *below* the newly inserted layer, leading to repeated coordinate 
            values.
    """
    
    assert np.all(np.abs(d[vert_dim].diff(dim=vert_dim)) == 1), ('Vert_dim index increments ' + 
                                                                 'must all be 1.')
    
    # To conserve nans in the original dataset, replace them with
    # fill_value in the coordinate array.
    assert not np.any(d[coords] == fill_value), 'dataset d contains fill_value.'
    d = d.where(np.logical_not(np.isnan(d[coords])), other=fill_value)
    
    below = d.where(d[coords] >= level[coords])
    above = d.where(d[coords] < level[coords])
       
    # Above the new coordinate, shift the vertical coordinate indices
    # up one.
    above = above.assign_coords({vert_dim: d[vert_dim] + 1})

    # Use broadcasting to fill regions below the new coordinate.
    out, _ = xarray.broadcast(below, above)

    # Fill regions above the new coordinate.
    above, _ = xarray.broadcast(above, out)
    out = above.where(np.isnan(out[coords]), other=out)
    
    # Any remaining nan values must now be the new level, so fill
    # those regions.
    new, _ = xarray.broadcast(level, out)
  
    # Subset to keys from new only.
    out = out[list(new.keys())]
    out = new.where(np.isnan(out[coords]), other=out)
    
    # Replace fill_value with nans.
    out = out.where(out != fill_value, other=np.nan)
        
    return out

def find_intersections(x, a, b, dim, log_x=False):
    """
    Find intersections of two lines that share x coordinates.
    
    Arguments:

        - x: The shared x coordinate values.
        - a: y values for line 1.
        - b: y values for line 2.
        - dim: The dimension along which the coordinates are indexed.
        - log_x: Use a logarithmic transform on x coordinates
          (e.g. for pressure coords)?
        
    Returns:

        - Dataset containing x, y coordinates for all intersections,
          increasing intersections and decreasing intersections. Note
          duplicates are not removed.
    """

    assert np.all(np.abs(x[dim].diff(dim=dim) == 1)), 'Index increments must all be 1.'
    
    if log_x:
        x = np.log(x)

    # Find intersections. Non-zero points in diffs indicates an
    # intersection.
    diffs = np.sign(a - b).diff(dim=dim)
    
    # Identify the points after each intersection.
    after_intersects = diffs.where(diffs == 0, other=1)
    
    # And the points just before each intersection.s
    before_intersects = xarray.concat([xarray.zeros_like(a.isel({dim: 0})), 
                                       after_intersects], dim=dim)
    before_intersects = before_intersects.shift({dim: -1}, fill_value=0)
    
    # The sign of the change for the intersect.
    sign_change = np.sign(a.where(after_intersects == 1) -
                          b.where(after_intersects == 1))

    x0 = x.where(before_intersects == 1).shift({dim: 1})
    x1 = x.where(after_intersects == 1)
    a0 = a.where(before_intersects == 1).shift({dim: 1})
    a1 = a.where(after_intersects == 1)
    b0 = b.where(before_intersects == 1).shift({dim: 1})
    b1 = b.where(after_intersects == 1) 

    # Calculate the x-intersection. This comes from finding the
    # equations of the two lines, one through (x0, a0) and (x1, a1)
    # and the other through (x0, b0) and (x1, b1), finding their
    # intersection, and reducing with a bunch of algebra.
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)

    # Calculate the y-intersection of the lines. Just plug the x above
    # into the equation for the line through the a points.
    intersect_y = ((intersect_x - x0) / (x1 - x0)) * (a1 - a0) + a0

    if log_x is True:
        intersect_x = np.exp(intersect_x)

    out = xarray.Dataset()
    out['all_intersect_x'] = intersect_x
    out['all_intersect_y'] = intersect_y
    out['increasing_x'] = intersect_x.where(sign_change > 0)
    out['increasing_y'] = intersect_y.where(sign_change > 0)
    out['decreasing_x'] = intersect_x.where(sign_change < 0)
    out['decreasing_y'] = intersect_y.where(sign_change < 0)
    out = out.rename({dim: 'offset_dim'})

    return out

def lfc_el(pressure, parcel_temperature, temperature, 
           lcl_pressure, lcl_temperature, vert_dim='model_level_number'):
    """
    Calculate the level of free convection (LFC) and equilibrium level
    (EL).
    
    Works by finding the first intersection of the ideal parcel path
    and the measured parcel temperature. If this intersection occurs
    below the LCL, the LFC is determined to be the same as the LCL,
    based upon the conditions set forth in [USAF1990]_, pg 4-14, where
    a parcel must be lifted dry adiabatically to saturation before it
    can freely rise.  The LFC returned is the 'bottom' LFC with
    highest pressure; the EL returned is the 'top' EL with the lowest
    pressure.

    Note that which temperature to use (virtual or real) is up to 
    the calling function.

    Arguments:

        - pressure: Pressure at each level [hPa].
        - parcel_temperature: Temperature of the parcel at each level [K].
        - temperature: Temperature of the environment at each level [K].
        - lcl_pressure: Pressure at the LCL [hPa].
        - lcl_temperature: Temperature at the LCL [K].
        - vert_dim: Vertical dimension name in input arrays.
    
    Returns:

        - DataArray with LFC pressure (lfc_pressure) and temperature
          (lfc_temperature).
    """
    
    # Find all intersections between parcel and environmental
    # temperatures by pressure.
    intersections = find_intersections(x=pressure,
                                       a=parcel_temperature, 
                                       b=temperature, 
                                       dim=vert_dim,
                                       log_x=True)

    # Find intersections again, ignoring first level.
    intersections_above = find_intersections(
        x=pressure.isel({vert_dim: slice(1,None)}),
        a=parcel_temperature.isel({vert_dim: slice(1,None)}),
        b=temperature.isel({vert_dim: slice(1,None)}),
        dim=vert_dim, log_x=True).reindex_like(intersections)
    
    # For points for which the atmosphere and parcel temperatures have
    # the same lowest-level value, ignore this point and find the real
    # LFC above it.
    intersections = intersections.where(
        (temperature.isel({vert_dim: 0}) !=
         parcel_temperature.isel({vert_dim: 0})),
        other=intersections_above)

    out = xarray.Dataset()
    
    # By default the first values are the lowest (highest pressure)
    # crossings for LFC and the highest (lowest pressure) crossings
    # for EL. The LFC also has to be above the LCL.
    above_lcl = intersections.increasing_x < lcl_pressure
        
    out['lfc_pressure'] = intersections.increasing_x.where(
        above_lcl).max(dim='offset_dim')
    out['lfc_temperature'] = intersections.increasing_y.where(
        intersections.increasing_x == out.lfc_pressure).max(dim='offset_dim')
    
    # Determine equilibrium pressure and temperature. The 'top'
    # (lowest pressure) EL is returned.
    out['el_pressure'] = intersections_above.decreasing_x.min(dim='offset_dim')
    out['el_temperature'] = intersections_above.decreasing_y.where(
        intersections.decreasing_x == out.el_pressure).max(dim='offset_dim')
    
    # If at the top of the atmosphere the parcel profile is warmer
    # than the environment, no EL exists. Also if EL is lower than or
    # equal to LCL, no EL exists.
    temps_available = np.logical_and(~np.isnan(parcel_temperature),
                                     ~np.isnan(temperature))
    top_pressure = pressure == pressure.where(temps_available).min(dim=vert_dim)
    top_prof_temp = parcel_temperature.where(top_pressure).max(dim=vert_dim)
    top_env_temp = temperature.where(top_pressure).max(dim=vert_dim)
    
    assert np.isnan(top_env_temp).broadcast_equals(np.isnan(temperature.max(dim=vert_dim))), 'Top temperature is NaN.'
    
    top_colder = top_prof_temp <= top_env_temp
    el_above_lcl = out.el_pressure < lcl_pressure
    el_exists = np.logical_and(top_colder, el_above_lcl)
    out['el_pressure'] = out.el_pressure.where(el_exists, other=np.nan)
    out['el_temperature'] = out.el_temperature.where(el_exists, other=np.nan)
    
    # There should only be one LFC and EL per point.
    assert not 'offset_dim' in out.keys(), 'Duplicate crossings detected.'

    # Identify points where no LFC intersections were found.
    lfc_missing = np.isnan(intersections.increasing_x.max(dim='offset_dim'))

    # If no intersection was found, but a parcel temperature above the
    # LCL is greater than the environmental temperature, return the
    # LCL.
    above_lcl = pressure < lcl_pressure
    pos_parcel = (parcel_temperature.where(above_lcl) >
                  temperature.where(above_lcl))
    pos_parcel = pos_parcel.any(dim=vert_dim)
    no_lfc_pos_parcel = np.logical_and(pos_parcel, lfc_missing)

    # Also return LCL if an intersection exists but all intersections
    # are below the LCL and EL is above the LCL.
    exists_but_na = np.logical_and(np.logical_not(lfc_missing),
                                   np.isnan(out.lfc_pressure))
    el_above_lcl = out.el_pressure < lcl_pressure
    lfc_below_el_above = np.logical_and(exists_but_na, el_above_lcl)
    
    # Do the replacements with LCL.
    replace_with_lcl = np.logical_or(no_lfc_pos_parcel, lfc_below_el_above)
    out['lfc_pressure'] = lcl_pressure.where(replace_with_lcl,
                                             other=out.lfc_pressure)
    out['lfc_temperature'] = lcl_temperature.where(
        replace_with_lcl,
        other=out.lfc_temperature)   
    
    # Assign metadata.
    out.el_pressure.attrs['long_name'] = 'Equilibrium level pressure'
    out.el_pressure.attrs['units'] = 'hPa'
    out.el_temperature.attrs['long_name'] = 'Equilibrium level temperature'
    out.el_temperature.attrs['units'] = 'K'
    out.lfc_pressure.attrs['long_name'] = 'Level of free convection pressure'
    out.lfc_pressure.attrs['units'] = 'hPa'
    out.lfc_temperature.attrs['long_name'] = ('Level of free convection ' +
                                              'temperature')
    out.lfc_temperature.attrs['units'] = 'K'

    return out

def trap_around_zeros(x, y, dim, log_x=True, start=0):
    """
    Calculate dx * y for points just before and after zeros in y.
    
    Arguments:

        - x: arrays of x along dim.
        - y: arrays of y along dim.
        - dim: Dimension along which to calculate.
        - log_x: Log transform x?
        - start: Zero-based position along dim to look for zeros.
        
    Returns:

        - a Dataset containing the areas and x coordinates for each
          rectangular area calculated before and after each zero; and
          an array of x coordinates that should be replaced by the new
          areas if integrating along x and including these areas
          afterwards.
    """
    
    assert np.all(np.abs(x[dim].diff(dim=dim)) == 1), 'Index increments must all be 1.'
    
    # Estimate zero crossings.
    zeros = xarray.zeros_like(y)
    zero_intersections = find_intersections(
        x=x.isel({dim: slice(start, None)}),
        a=y.isel({dim: slice(start, None)}),
        b=zeros.isel({dim: slice(start, None)}),
        dim=dim, log_x=log_x)
    zero_intersections = zero_intersections.rename({'offset_dim': dim})
    zero_y = zero_intersections.all_intersect_y
    zero_x = zero_intersections.all_intersect_x
        
    # Take log of x if required.
    if log_x:
        x = np.log(x)
        zero_x = np.log(zero_x)
    
    zero_level = xarray.zeros_like(y.isel({dim: 0}))
    
    after_zeros_mask = np.logical_not(np.isnan(zero_y))
    before_zeros_mask = xarray.concat([zero_level, zero_y],
                                      dim=dim).shift({dim: -1})
    before_zeros_mask = np.logical_not(np.isnan(before_zeros_mask))
    
    def calc_areas(x, y, mask, shift_x=0):
        areas = xarray.Dataset({'area': xarray.zeros_like(mask),
                                'dx': xarray.zeros_like(mask),
                                'x': xarray.zeros_like(mask)})

        # Get coordinates of zeros.
        x_near_zero = x.where(mask)
        y_near_zero = y.where(mask)
        
        # Determine the value of y (mean of y and zero) and dx.
        mean_y = y_near_zero / 2
        
        dx = x_near_zero.shift({dim:shift_x}) - zero_x
        dx = xarray.concat([zero_level, dx], dim=dim).shift({dim: -shift_x})
    
        areas['area'] = mean_y * np.abs(dx)
        areas['x'] = x_near_zero - dx/2
        areas['dx'] = np.abs(dx)
        
        if x.chunks is not None:
            areas = areas.chunk(200)
            areas = areas.chunk({dim: areas[dim].size})
            
        areas = areas.reset_coords(drop=True)
        return(areas)
        
    areas_before_zeros = calc_areas(x=x, y=y, mask=before_zeros_mask, shift_x=1)
    areas_after_zeros = calc_areas(x=x, y=y, mask=after_zeros_mask, shift_x=0)
   
    # Concatenate areas before zeros and areas after zeros.
    areas = xarray.concat([areas_before_zeros, areas_after_zeros], dim=dim)
    
    # Determine start/end points on x axis for each area.
    areas['x_from'] = areas.x - areas.dx/2
    areas['x_to'] = areas.x + areas.dx/2
    
    # Mask is a mask that selects elements that were *not* included in
    # the differences; to be used by a CAPE calculation where we don't
    # want to count the areas around zeros twice.
    mask = xarray.full_like(x, True)
    mask, bef = xarray.broadcast(mask, areas_before_zeros)
    mask = mask.where(np.isnan(bef.area), other=False)
    
    return areas, mask
   
def cape_cin_base(pressure, temperature, lfc_pressure, el_pressure,
                  parcel_temperature, vert_dim='model_level_number',
                  pos_cape_neg_cin=True, post_zero_cin=False, **kwargs):
    """
    Calculate CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and
    convective inhibition (CIN) of a given upper air profile and
    parcel path. CIN is integrated between the surface and LFC, CAPE
    is integrated between the LFC and EL (or top of
    sounding). Intersection points of the measured temperature profile
    and parcel profile are logarithmically interpolated.
    
    Uses the bottom (highest-pressure) LFC and the top
    (lowest-pressure) EL.

    Arguments:

        - pressure: Pressure level(s) of interest [hPa].
        - temperature: Temperature at each pressure level [K].
        - lfc_pressure: Pressure of level of free convection [hPa].
        - el_pressure: Pressure of equilibrium level [hPa].
        - parcel_temperature: The temperature of the lifted parcel.
        - vert_dim: The vertical dimension.
        - pos_cape_neg_cin: Force CAPE to be positive and CIN to be negative by
                            counting only positive (negative) buoyance for 
                            CAPE (CIN).
        - post_zero_cin: Reset any positive CIN values to zero?

    Returns:

        - Dataset with convective available potential energy (cape)
          and convective inhibition (cin), both in J kg-1.

    """

    # Where the EL is nan, use the highest (lowest-pressure) value as
    # the EL.
    el_pressure = pressure.min(dim=vert_dim).where(np.isnan(el_pressure), 
                                                   other=el_pressure)

    # Difference between the parcel path and measured temperature
    # profiles.
    temp_diffs = xarray.Dataset({'temp_diff': (parcel_temperature -
                                               temperature),
                                 'pressure': pressure,
                                 'log_pressure': np.log(pressure)})
    
    # Integration areas around zero differences. Note MetPy's
    # implementation in _find_append_zero_crossings() looks for
    # intersections from the 2nd index onward (start=1 in this code);
    # but in this implementation the whole array needs to be checked
    # (start=0) for the unit tests to pass.
    areas_around_zeros, trapz_mask = trap_around_zeros(x=temp_diffs.pressure, 
                                                       y=temp_diffs.temp_diff,  
                                                       dim=vert_dim, log_x=True)
    areas_around_zeros['x'] = np.exp(areas_around_zeros.x)
    areas_around_zeros['x_from'] = np.exp(areas_around_zeros.x_from)
    areas_around_zeros['x_to'] = np.exp(areas_around_zeros.x_to)
     
    # Integrate positive areas between LFC and EL pressure levels to get CAPE.
    diffs_lfc_to_el = temp_diffs.where(pressure <= lfc_pressure)
    diffs_lfc_to_el = diffs_lfc_to_el.where(pressure >= el_pressure)
    areas_lfc_to_el = areas_around_zeros.where(areas_around_zeros.x <=
                                               lfc_pressure)
    areas_lfc_to_el = areas_lfc_to_el.where(areas_around_zeros.x >= el_pressure)
    
    if pos_cape_neg_cin:
        areas_lfc_to_el = areas_lfc_to_el.where(areas_lfc_to_el.area > 0)
    
    cape = mpconsts.Rd.m * trapz(dat=diffs_lfc_to_el, x='log_pressure', 
                                 dim=vert_dim, mask=trapz_mask, 
                                 only_positive=pos_cape_neg_cin)
    cape = cape.reset_coords().temp_diff
    cape = cape + (mpconsts.Rd.m * areas_lfc_to_el.area.sum(dim=vert_dim))
    cape.name = 'cape'
    cape.attrs['long_name'] = 'Convective available potential energy'
    cape.attrs['units'] = 'J kg$^{-1}$'

    # Integrate negative between surface and LFC to get CIN.
    temp_diffs_surf_to_lfc = temp_diffs.where(pressure >= lfc_pressure)
    areas_surf_to_lfc = areas_around_zeros.where(areas_around_zeros.x >=
                                                 lfc_pressure)
    
    if pos_cape_neg_cin:
        areas_surf_to_lfc = areas_surf_to_lfc.where(areas_surf_to_lfc.area < 0)
    
    cin = mpconsts.Rd.m * trapz(dat=temp_diffs_surf_to_lfc, x='log_pressure', 
                                dim=vert_dim, mask=trapz_mask, 
                                only_negative=pos_cape_neg_cin)
    cin = cin.reset_coords().temp_diff
    cin = cin + (mpconsts.Rd.m * areas_surf_to_lfc.area.sum(dim=vert_dim))
    cin.name = 'cin'
    cin.attrs['long_name'] = 'Convective inhibition'
    cin.attrs['units'] = 'J kg$^{-1}$'

    if post_zero_cin:
        cin = cin.where(cin <= 0, other=0)
    
    res = xarray.merge([cape, cin])
    res.attrs = []
    return res

def cape_cin(pressure, temperature, dewpoint, parcel_temperature, parcel_pressure,
             parcel_dewpoint, vert_dim='model_level_number', 
             virtual_temperature_correction=True, lcl_interp='log',
             **kwargs):
    """
    Calculate CAPE and CIN; wraps finding of LFC and parcel profile
    and call to cape_cin_base. Uses the bottom (highest-pressure) LFC
    and the top (lowest-pressure) EL. The virtual temperature correction is
    optional; it defaults to False to match MetPy's implementation, but is
    it is recommended as a more correct way to calculate CAPE/CIN.

    Arguments:

        - pressure: Pressure level(s) of interest [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpont at each pressure level [K].
        - parcel_temperature: The temperature of the starting parcel [K].
        - parcel_pressure: The pressure of the starting parcel [K].
        - parcel_dewpoint: The dewpoint of the starting parcel [K].
        - vert_dim: The vertical dimension.
        - lcl_interp: Interpolator for lcl environment (linear or log).
        - **kwargs: Optional extra arguments to cape_cin_base.

    Returns:

        - Dataset with convective available potential energy (cape)
          and convective inhibition (cin), both in J kg-1, plus the
          lifted profile.
    """
    
    # Calculate parcel profile. The LCL is always calculated using real temperature.
    profile = parcel_profile_with_lcl(pressure=pressure,
                                      temperature=temperature,
                                      dewpoint=dewpoint,
                                      parcel_temperature=parcel_temperature,
                                      parcel_pressure=parcel_pressure,
                                      parcel_dewpoint=parcel_dewpoint,
                                      vert_dim=vert_dim,
                                      lcl_interp=lcl_interp)
    
    # Apply the virtual temperature correction?
    # By default, MetPy does not use any virtual temperature correction.
    if not virtual_temperature_correction:
        # Calculate LFC and EL.
        parcel_lfc_el = lfc_el(pressure=profile.pressure,
                               parcel_temperature=profile.temperature, 
                               temperature=profile.environment_temperature, 
                               lcl_pressure=profile.lcl_pressure, 
                               lcl_temperature=profile.lcl_temperature,
                               vert_dim=vert_dim)

        # Calculate CAPE and CIN.
        cape_cin = cape_cin_base(pressure=profile.pressure,
                                 temperature=profile.environment_temperature, 
                                 lfc_pressure=parcel_lfc_el.lfc_pressure, 
                                 el_pressure=parcel_lfc_el.el_pressure, 
                                 parcel_temperature=profile.temperature,
                                 vert_dim=vert_dim, **kwargs)
        
        cape_cin.attrs['correction'] = ('Virtual temperature correction not used ' + 
                                        'in CAPE/CIN calculations.')
    else:
        # Calculate LFC and EL.
        parcel_lfc_el = lfc_el(pressure=profile.pressure,
                               parcel_temperature=profile.virtual_temperature, 
                               temperature=profile.environment_virtual_temperature, 
                               lcl_pressure=profile.lcl_pressure, 
                               lcl_temperature=profile.lcl_virtual_temperature,
                               vert_dim=vert_dim)

        # Calculate CAPE and CIN.
        cape_cin = cape_cin_base(pressure=profile.pressure,
                                 temperature=profile.environment_virtual_temperature, 
                                 lfc_pressure=parcel_lfc_el.lfc_pressure, 
                                 el_pressure=parcel_lfc_el.el_pressure, 
                                 parcel_temperature=profile.virtual_temperature,
                                 vert_dim=vert_dim, **kwargs)
        
        cape_cin.attrs['correction'] = ('Virtual temperature correction used ' + 
                                        'in CAPE/CIN calculations.')
    
    return cape_cin, xarray.merge([profile, parcel_lfc_el])
    
def surface_based_cape_cin(pressure, temperature, dewpoint,
                           vert_dim='model_level_number',
                           prefix=None, **kwargs):
    """
    Calculate surface-based CAPE and CIN.

    Arguments:

        - pressure: Pressure level(s) of interest [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint at each level [K].
        - vert_dim: The vertical dimension.
        - prefix: Variable prefix.
        - **kwargs: Optional extra arguments to cape_cin.
        
    Returns:

        - Dataset with convective available potential energy (cape) and 
          convective inhibition (cin), both in J kg-1, plus the lifted profile.
    """
    
    # Profile for surface-based parcel ascent.
    res, profile = cape_cin(pressure=pressure,
                            temperature=temperature,
                            dewpoint=dewpoint,
                            parcel_temperature=temperature.isel({vert_dim: 0}),
                            parcel_pressure=pressure.isel({vert_dim: 0}),
                            parcel_dewpoint=dewpoint.isel({vert_dim: 0}),
                            vert_dim=vert_dim,
                            **kwargs)
    
    res.cape.attrs['description'] = 'CAPE for surface-based parcel.'
    res.cin.attrs['description'] = 'CIN for surface-based parcel.'
    if not prefix is None:
        res = res.rename({'cape': prefix+'_cape',
                          'cin': prefix+'_cin'})
    
    return res, profile
        

def from_most_unstable_parcel(pressure, temperature, dewpoint,
                              vert_dim='model_level_number', depth=300):
    """
    Select pressure and temperature data at and above the most unstable
    parcel within the first x hPa above the surface.
    
    Arguments:

        - pressure: Pressure level(s) of interest [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint at each level [K].
        - vert_dim: The vertical dimension.
        - depth: The depth above the surface (lowest-level pressure)
                 in which to look for the most unstable parcel.
        
    Returns:

        - subset pressure, subset temperature, subset dewpoint, 
          and most unstable layer.
    """
    
    assert pressure.name == 'pressure', 'Pressure requires name pressure.'
    assert temperature.name == 'temperature', ('Temperature requires ' +
                                               'name temperature.')
    assert dewpoint.name == 'dewpoint', 'Dewpoint requires name dewpoint.'
        
    dat = xarray.merge([pressure, temperature, dewpoint])
    dat.attrs = []
        
    # Find the most unstable layer in the lowest 'depth' hPa.
    unstable_layer = most_unstable_parcel(dat=dat, depth=depth, 
                                          vert_dim=vert_dim)
        
    # Subset to layers at or above the most unstable parcels.
    dat = dat.where(pressure <= unstable_layer.pressure)
    dat = dat.dropna(dim=vert_dim, how='all')
    dat = shift_out_nans(x=dat, name='pressure', dim=vert_dim)
    
    return dat.pressure, dat.temperature, dat.dewpoint, unstable_layer

def most_unstable_cape_cin(pressure, temperature, dewpoint,
                           vert_dim='model_level_number', depth=300,
                           prefix=None, **kwargs):
    """
    Calculate CAPE and CIN for the most unstable parcel within a given 
    depth above the surface..

    Arguments:

        - pressure: Pressure level(s) of interest [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint at each level [K].
        - vert_dim: The vertical dimension.
        - depth: The depth above the surface (lowest-level pressure)
                 in which to look for the most unstable parcel.
        - prefix: Prefix for variable names.
        - **kwargs: Optional extra arguments to cape_cin.
        
    Returns:

        - Dataset with convective available potential energy (cape) and 
          convective inhibition (cin), both in J kg-1, plus the 
          lifted profile and the lifted parcel.
    """
    
    pressure, temperature, dewpoint, unstable_layer = from_most_unstable_parcel(
        pressure=pressure, temperature=temperature, dewpoint=dewpoint,
        vert_dim=vert_dim, depth=depth)
        
    res, profile = cape_cin(pressure=pressure,
                            temperature=temperature,
                            dewpoint=dewpoint,
                            parcel_temperature=unstable_layer.temperature, 
                            parcel_pressure=unstable_layer.pressure,
                            parcel_dewpoint=unstable_layer.dewpoint,   
                            vert_dim=vert_dim,
                            **kwargs)

    desc = f'most-unstable parcel in lowest {depth} hPa.'
    res.cape.attrs['description'] = f'CAPE for {desc}'
    res.cin.attrs['description'] = f'CIN for {desc}'
    if not prefix is None:
        res = res.rename({'cape': prefix+'_cape',
                          'cin': prefix+'_cin'})
    
    return res, profile, unstable_layer
        
def mix_layer(pressure, temperature, dewpoint, vert_dim='model_level_number', 
              depth=100, load=True):
    """
    Fully mix the lowest x hPa in vertical profiles.
    
    Arguments:
        
        - pressure: Pressure level(s) [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint at each level [K].
        - vert_dim: The vertical dimension.
        - depth: The depth above the surface (lowest-level pressure)
          to mix [hPa].
        - load: Load resulting parcel into memory?
    
    Returns:
    
        - pressure, temperature, and dewpoint profiles and 
          the mixed layer parcel.
    """
    
    # Mix the lowest x hPa.
    mp = mixed_parcel(pressure=pressure, temperature=temperature, 
                      dewpoint=dewpoint, depth=depth, vert_dim=vert_dim)
    
    # Remove layers that were part of the mixed layer. 
    assert pressure.name == 'pressure', 'Pressure requires name pressure.'
    assert temperature.name == 'temperature', ('Temperature requires ' +
                                               'name temperature.')
    assert dewpoint.name == 'dewpoint', 'Dewpoint requires name dewpoint.'
        
    dat = xarray.merge([pressure, temperature, dewpoint])
    dat = dat.where(pressure < (pressure.max(dim=vert_dim) - depth))
    dat = dat.dropna(dim=vert_dim, how='all')
    dat = shift_out_nans(x=dat, name='pressure', dim=vert_dim)
    
    # Add the mixed layer to the bottom of the profiles.
    mp[vert_dim] = dat.pressure[vert_dim].min() - 1
    pressure = xarray.concat([mp.pressure, dat.pressure], dim=vert_dim)
    temperature = xarray.concat([mp.temperature, dat.temperature], dim=vert_dim)
    dewpoint = xarray.concat([mp.dewpoint, dat.dewpoint], dim=vert_dim)
    
    if load:
        mp = mp.load()
    
    return pressure, temperature, dewpoint, mp
    
def mixed_layer_cape_cin(pressure, temperature, dewpoint, vert_dim='model_level_number',
                         depth=100, prefix=None, **kwargs):
    """
    Calculate CAPE and CIN for a fully-mixed lowest x hPa parcel.

    Arguments:

        - pressure: Pressure level(s) of interest [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint at each level [K].
        - vert_dim: The vertical dimension.
        - depth: The depth above the surface (lowest-level pressure)
          to mix [hPa].
        - prefix: Variable name prefix.
        - **kwargs: Optional extra arguments to cape_cin.
        
    Returns:

        - Dataset with convective available potential energy (cape)
          and convective inhibition (cin), both in J kg-1, plus the
          lifted profile and mixed parcel.
    """

    pressure, temperature, dewpoint, mp = mix_layer(pressure=pressure,
                                                    temperature=temperature,
                                                    dewpoint=dewpoint,
                                                    vert_dim=vert_dim,
                                                    depth=depth)

    res, profile = cape_cin(pressure=pressure,
                            temperature=temperature,
                            dewpoint=dewpoint,
                            parcel_temperature=mp.temperature, 
                            parcel_pressure=mp.pressure,
                            parcel_dewpoint=mp.dewpoint,
                            vert_dim=vert_dim,
                            **kwargs)

    desc = f'fully-mixed lowest {depth} hPa parcel'
    res.cape.attrs['description'] = f'CAPE for {desc}.'
    res.cin.attrs['description'] = f'CIN for {desc}'
    
    if not prefix is None:
        res = res.rename({'cape': prefix+'_cape',
                          'cin': prefix+'_cin'})
    
    return res, profile, mp
    
def shift_out_nans(x, name, dim):
    """
    Shift data along a dim to remove all leading nans in that
    dimension, element-wise.
    
    Arguments:

        - x: The data to work on.
        - name: The name within data in which to look for nans.
        - dim: The dimension to shift.

    """
    
    assert np.all(np.abs(x[dim].diff(dim=dim)) == 1), 'Index increments must all be 1.'
    
    for i in np.arange(len(x[dim])):
        if not np.any(np.isnan(x[name].isel({dim: 0}))):
            break
        shifted = x.shift({dim: -1})
        x = shifted.where(np.isnan(x[name].isel({dim: 0})), other=x)
        
    return x

def lifted_index(profile, vert_dim='model_level_number', description=None,
                 prefix=None):
    """
    Calculate the lifted index. 
    
    Lifted index formula derived from Galway 1956 and referenced by
    DoswellSchultz 2006.

    Arguments:

        - profile: Profile as returned by parcel_profile_with_lcl().
        - vert_dim: The vertical dimension name.
        - description: Description to add.
        - prefix: Variable prefix to use.

    Returns:

        - Lifted index at each point [K].
    """
    
    # Interpolate to get 500 hPa values.
    dat = log_interp(x=profile, coords=profile.pressure, at=500, dim=vert_dim)
    dat = dat.reset_coords(drop=True)
    
    # Calculate lifted index.
    li = xarray.Dataset({'lifted_index': (dat.environment_temperature -
                                          dat.temperature)})
    li.lifted_index.attrs['long_name'] = 'Lifted index'
    li.lifted_index.attrs['units'] = 'K'
    if not description is None:
        li.lifted_index.attrs['description'] = description
    if not prefix is None:
        li = li.rename({'lifted_index': prefix+'_lifted_index'})
    
    return li
    
def linear_interp(x, coords, at, dim='model_level_number', keep_attrs=True, extrapolate=False):
    """
    Perform simple linear interpolation.
    
    Arguments:
    
        - x: Data set to interpolate.
        - coords: Coordinate value for each point in x.
        - at: Points at which to interpolate.
        - dim: The dimension along which to interpolate.
        - keep_attrs: Keep attributes?
        - extrapolate: Allow extrapolation for out of range points? (Default false since 
                       linear interp innacurate with large extrapolations).
    """
    
    # Coordinate values before and after (below and above?) the interp coord.
    coords_before = coords.where(coords >= at).min(dim=dim)
    coords_after = coords.where(coords <= at).max(dim=dim)

    if extrapolate:
        # Nans in coords_before are points where we should extrapolate below the coordinate range.
        # Nans in coords_after are points where we should extrapolate above the coordinate range.
        extrap_below = np.isnan(coords_before)
        extrap_above = np.isnan(coords_after)

        if np.any(extrap_below) or np.any(extrap_above):
            # Use the second lowest/highest points to determine extrapolation slopes.
            second_lowest = coords.where(coords != coords.max(dim=dim)).max(dim=dim)
            second_highest = coords.where(coords != coords.min(dim=dim)).min(dim=dim)

            # Note that duplicate min/max coordinate values are ignored here.
            coords_before = coords_after.where(extrap_below, other=coords_before)
            coords_after = second_lowest.where(extrap_below, other=coords_after)

            coords_after = coords_before.where(extrap_above, other=coords_after)
            coords_before = second_highest.where(extrap_above, other=coords_before)
            assert len(np.unique(np.sign(coords_before - coords_after)) == 1), 'Extrapolation error.'
    
    # If there are duplicate coordinates, take the mean of the values 
    # for those coordinates.
    x_before = x.where(coords == coords_before).mean(dim=dim)
    x_after = x.where(coords == coords_after).mean(dim=dim)
    
    # The interpolated values.
    res = x_before + (x_after - x_before) * ((at - coords_before) /
                                             (coords_after - coords_before))
    
    # When the interpolated x exists already, return it.
    res = x_before.where(x_before == x_after, other=res)

    if keep_attrs:
        res.attrs = x.attrs
    
    return(res)

def log_interp(x, coords, at, dim='model_level_number'):
    """
    Run linear_interp on logged coordinate values.
    
    Arguments:
        - x: Data set to interpolate.
        - coords: Coordinate value for each point in x.
        - at: Points at which to interpolate.
        - dim: The dimension along which to interpolate.
    
    It is assumed that x[coords] is sorted. Note that if coordinates themselves 
    are also interpolated, the returned coordinate values will not equal the 
    value of 'at'.
    """
    
    return linear_interp(x=x, coords=np.log(coords), at=np.log(at), dim=dim)
    
def deep_convective_index(pressure, temperature, dewpoint, lifted_index, 
                          vert_dim='model_level_number', description=None, 
                          prefix=None):
    """
    Calculate the deep convective index (DCI) as defined by Kunz 2009.
    
    Arguments:

        - pressure: Pressure values [hPa].
        - temperature: Temperature at each pressure [K].
        - dewpoint: Dewpoint temperature at each pressure.
        - lifted_index: The lifted index.
        - vert_dim: The vertical dimension name.
        - description: Description to add to variable.
        - prefix: Prefix to use for variable name.
    
    Returns:

        - The DCI [C] per point.
    """
    
    dat = xarray.merge([pressure, temperature, dewpoint])
    
    # Interpolate to get 850 hPa values.
    dat = log_interp(x=dat, coords=dat.pressure, at=850, dim=vert_dim)
    dat = dat.reset_coords(drop=True)
    
    # Convert temperature and dewpoint from K to C.
    dat['temperature'] = dat.temperature - 273.15
    dat['dewpoint'] = dat.dewpoint - 273.15
    
    dci = xarray.Dataset({'dci': dat.temperature + dat.dewpoint - lifted_index})
    dci.dci.attrs['long_name'] = 'Deep convective index'
    dci.dci.attrs['units'] = 'C'
    
    if not description is None:
        dci.dci.attrs['description'] = description
    if not prefix is None:
        dci = dci.rename({'dci': prefix+'_dci'})
    
    return dci
    
def min_conv_properties(dat, vert_dim='model_level_number'):
    """
    Calculate a minimal set of convective properties for a set of points. 
    
    Arguments:
    
       - dat: An xarray Dataset containing pressure, temperature, and 
              specific humidity, wind data, and height ('height' for all 
              variables except wind, 'wind_height' for wind levels).
       - vert_dim: The name of the vertical dimension in the dataset.
            
    Returns:
    
        - Dataset containing convection properties for each point.
    """

    print('Calculating dewpoint...')
    dat['dewpoint'] = metpy.calc.dewpoint_from_specific_humidity(
        pressure=dat.pressure,
        temperature=dat.temperature,
        specific_humidity=dat.specific_humidity)
    dat['dewpoint'] = dat.dewpoint.metpy.convert_units('K')
    dat['dewpoint'] = dat.dewpoint.metpy.dequantify()
    
    print('Calculating mixed-parcel CAPE and CIN (100 hPa)...')
    mixed_cape_cin_100, mixed_profile_100, _ = mixed_layer_cape_cin(
        pressure=dat.pressure,
        temperature=dat.temperature, 
        dewpoint=dat.dewpoint,
        vert_dim=vert_dim,
        prefix='mixed_100')
    
    print('Calculating lifted indices...')
    mixed_li_100 = lifted_index(profile=mixed_profile_100, vert_dim=vert_dim, 
                                prefix='mixed_100', 
                                description=('Lifted index using fully-mixed ' + 
                                             'lowest 100 hPa parcel.'))
    
    print('700-500 hPa lapse rate...')
    lapse = lapse_rate(pressure=dat.pressure, 
                       temperature=dat.temperature, 
                       height=dat.height_asl,
                       vert_dim=vert_dim,)
    lapse.name = 'lapse_rate_700_500'

    print('Temperature at 500 hPa...')
    temp_500 = isobar_temperature(pressure=dat.pressure, 
                                  temperature=dat.temperature, 
                                  isobar=500, 
                                  vert_dim=vert_dim)
    temp_500.name = 'temp_500'
    
    print('Freezing level height...')
    flh = freezing_level_height(temperature=dat.temperature,
                                height=dat.height_asl,
                                vert_dim=vert_dim)
    
    print('Melting level height...')
    mlh, _ = melting_level_height(pressure=dat.pressure,
                                  temperature=dat.temperature,
                                  dewpoint=dat.dewpoint,
                                  height=dat.height_asl,
                                  vert_dim=vert_dim)
    
    print('0-6 km vertical wind shear...')
    shear = wind_shear(surface_wind_u=dat.surface_wind_u, 
                       surface_wind_v=dat.surface_wind_v, 
                       wind_u=dat.wind_u, 
                       wind_v=dat.wind_v, 
                       height=dat.wind_height_above_surface, 
                       shear_height=6000, 
                       vert_dim=vert_dim)

    print('Merging results...')
    out = xarray.merge([mixed_cape_cin_100, mixed_li_100, 
                        lapse, temp_500, flh, mlh, shear])
    
    return out
     
def conv_properties(dat, vert_dim='model_level_number', ignore_nans=False):
    """
    Calculate selected convective properties for a set of points. 
    
    Arguments:
    
       - dat: An xarray Dataset containing pressure, temperature, and 
              specific humidity, wind data, and height ('height' for all 
              variables except wind, 'wind_height' for wind levels).
       - vert_dim: The name of the vertical dimension in the dataset.
       - ignore_nans: Only return values where there are no nans?
            
    Returns:
    
        - Dataset containing convection properties for each point.
    """

    print('Calculating dewpoint...')
    dat['dewpoint'] = metpy.calc.dewpoint_from_specific_humidity(
        pressure=dat.pressure,
        temperature=dat.temperature,
        specific_humidity=dat.specific_humidity)
    dat['dewpoint'] = dat.dewpoint.metpy.convert_units('K')
    dat['dewpoint'] = dat.dewpoint.metpy.dequantify()

    valid_points = np.logical_and(~np.isnan(dat.dewpoint).any(vert_dim),
                                  ~np.isnan(dat.pressure).any(vert_dim))
    valid_points = np.logical_and(valid_points,
                                  ~np.isnan(dat.temperature).any(vert_dim))
    valid_points = np.logical_and(valid_points,
                                  ~np.isnan(dat.specific_humidity).any(vert_dim))

    print('Calculating most-unstable CAPE and CIN...')
    mu_cape_cin, mu_profile, mu_parcel = most_unstable_cape_cin(
        pressure=dat.pressure,
        temperature=dat.temperature, 
        dewpoint=dat.dewpoint,
        vert_dim=vert_dim,
        depth=250, prefix='mu')

    print('Calculating mixed-parcel CAPE and CIN (100 hPa)...')
    mixed_cape_cin_100, mixed_profile_100, _ = mixed_layer_cape_cin(
        pressure=dat.pressure,
        temperature=dat.temperature, 
        dewpoint=dat.dewpoint,
        vert_dim=vert_dim,
        prefix='mixed_100')
    
    print('Calculating mixed-parcel CAPE and CIN (50 hPa)...')
    mixed_cape_cin_50, mixed_profile_50, _ = mixed_layer_cape_cin(
        pressure=dat.pressure,
        temperature=dat.temperature, 
        dewpoint=dat.dewpoint,
        vert_dim=vert_dim,
        depth=50,
        prefix='mixed_50')
    
    print('Calculating lifted indices...')
    mu_li = lifted_index(profile=mu_profile, vert_dim=vert_dim, prefix='mu',
                         description=('Lifted index using most-unstable ' + 
                                      'parcel in lowest 250 hPa.'))
    mixed_li_100 = lifted_index(profile=mixed_profile_100, vert_dim=vert_dim, 
                                prefix='mixed_100', 
                                description=('Lifted index using fully-mixed ' + 
                                             'lowest 100 hPa parcel.'))
    mixed_li_50 = lifted_index(profile=mixed_profile_50, vert_dim=vert_dim,
                               prefix='mixed_50', 
                               description=('Lifted index using fully-mixed ' + 
                                            'lowest 50 hPa parcel.'))
    
    print('Calculating deep convective indices...')
    mu_dci = deep_convective_index(pressure=dat.pressure, 
                                   temperature=dat.temperature,
                                   dewpoint=dat.dewpoint, 
                                   lifted_index=mu_li.mu_lifted_index,
                                   vert_dim=vert_dim,
                                   prefix='mu',
                                   description=('Deep convective index using most-unstable ' + 
                                                'parcel in lowest 250 hPa.'))
    mixed_dci_100 = deep_convective_index(pressure=dat.pressure, 
                                          temperature=dat.temperature,
                                          dewpoint=dat.dewpoint, 
                                          lifted_index=mixed_li_100.mixed_100_lifted_index,
                                          vert_dim=vert_dim,
                                          prefix='mixed_100',
                                          description=('Deep convective index using fully-mixed ' + 
                                                       'lowest 100 hPa parcel.'))
    mixed_dci_50 = deep_convective_index(pressure=dat.pressure, 
                                         temperature=dat.temperature,
                                         dewpoint=dat.dewpoint, 
                                         lifted_index=mixed_li_50.mixed_50_lifted_index,
                                         vert_dim=vert_dim,
                                         prefix='mixed_50',
                                         description=('Deep convective index using fully-mixed ' + 
                                                      'lowest 50 hPa parcel.'))
    
    print('Calculating mixing ratio of most unstable parcel...')
    mu_mixing_ratio = metpy.calc.mixing_ratio_from_specific_humidity(
        specific_humidity=metpy.calc.specific_humidity_from_dewpoint(
            pressure=mu_parcel.pressure*units.hPa,
            dewpoint=mu_parcel.dewpoint*units.K)).metpy.dequantify()
    mu_mixing_ratio.attrs['long_name'] = 'Mixing ratio'
    mu_mixing_ratio.attrs['description'] = 'Mixing ratio of most unstable parcel'
    mu_mixing_ratio.name = 'mu_mixing_ratio'
    
    print('700-500 hPa lapse rate...')
    lapse = lapse_rate(pressure=dat.pressure, 
                       temperature=dat.temperature, 
                       height=dat.height_asl,
                       vert_dim=vert_dim,)
    lapse.name = 'lapse_rate_700_500'

    print('Temperature at 500 hPa...')
    temp_500 = isobar_temperature(pressure=dat.pressure, 
                                  temperature=dat.temperature, 
                                  isobar=500, 
                                  vert_dim=vert_dim)
    temp_500.name = 'temp_500'

    print('Freezing level height...')
    flh = freezing_level_height(temperature=dat.temperature, 
                                height=dat.height_asl,
                                vert_dim=vert_dim)
    
    print('Melting level height...')
    mlh, _ = melting_level_height(pressure=dat.pressure,
                                  temperature=dat.temperature,
                                  dewpoint=dat.dewpoint,
                                  height=dat.height_asl,
                                  vert_dim=vert_dim)

    print('0-6 km vertical wind shear...')
    shear = wind_shear(surface_wind_u=dat.surface_wind_u, 
                       surface_wind_v=dat.surface_wind_v, 
                       wind_u=dat.wind_u, 
                       wind_v=dat.wind_v, 
                       height=dat.wind_height_above_surface, 
                       shear_height=6000, 
                       vert_dim=vert_dim)

    print('Merging results...')
    out = xarray.merge([mu_cape_cin, mu_mixing_ratio, 
                        mixed_cape_cin_100, mixed_cape_cin_50, 
                        mu_li, mixed_li_100, mixed_li_50, 
                        mu_dci, mixed_dci_100, mixed_dci_50, lapse, 
                        temp_500, flh, mlh, shear])
    
    if not ignore_nans:
        out = out.where(valid_points, other=np.nan)
    return out
        
def lapse_rate(pressure, temperature, height, from_pressure=700, to_pressure=500, 
               vert_dim='model_level_number'):
    """
    Calculate the observed environmental lapse rate between two pressure levels.
    
    Arguments:
        
        - pressure: Pressure at each level [hPa].
        - temperature: Temperature at each level [K].
        - height: Height of each level [m]
        - from_pressure: Pressure level to calculate from [hPa].
        - to_pressure: Pressure level to calculate to [hPa].
        - vert_dim: Name of vertical dimension.
        
    Returns:
    
        - Lapse rate between two levels at each point [K km-1].
    """
    
    from_temperature = log_interp(x=temperature, coords=pressure, 
                                  at=from_pressure, dim=vert_dim)
    to_temperature = log_interp(x=temperature, coords=pressure, 
                                at=to_pressure, dim=vert_dim)
    from_height = log_interp(x=height, coords=pressure, 
                             at=from_pressure, dim=vert_dim)/1000
    to_height = log_interp(x=height, coords=pressure, 
                           at=to_pressure, dim=vert_dim)/1000
        
    lapse = (to_temperature - from_temperature) / (to_height - from_height)
    lapse.attrs['long_name'] = 'Lapse rate'
    lapse.attrs['description'] = f'{from_pressure}-{to_pressure} hPa lapse rate'
    lapse.attrs['units'] = 'K km$^{-1}$'
    
    return lapse

def freezing_level_height(temperature, height, vert_dim='model_level_number'):
    """
    Calculate the freezing level height by looking for 0 degrees in a temperature field.
    
    Arguments:
    
        - temperature: Temperature at each level [K]. Assumed to be dry-bulb temperature;
                       use melting_level_height to add calculation of wet-bulb temperature.
        - height: Height of each level [m].
        - vert_dim: Name of vertical dimension.
        
    Returns:
    
        - Freezing level height [m].
    """
    
    _, zeros = xarray.broadcast(temperature, xarray.DataArray(273.15))
    intersects = find_intersections(x=height, a=temperature, b=zeros, dim=vert_dim)
    flh = intersects.all_intersect_x.min(dim='offset_dim')
    flh.attrs['long_name'] = f'Freezing-level height'
    flh.attrs['description'] = f'Height of zero degree dry-bulb temperature isotherm.'
    flh.attrs['units'] = 'm'
    flh.name = 'freezing_level'
    return flh

def melting_level_height(pressure, temperature, dewpoint, height, fast=True, vert_dim='model_level_number'):
    """    
    Calculate the melting level height by looking for 0 degrees in the wet-bulb temperature field.
    
    Arguments:
        - pressure: Pressure level(s) of interest [hPa].
        - temperature: Temperature at each pressure level [K].
        - dewpoint: Dewpoint temperatures [K].
        - height: Height of each level [m].
        - vert_dim: The vertical dimension to operate on.
        
    Returns:
    
        - Melting-level height [m], and the wet bulb temperature of each point.
    """
    
    print('Calculating wet bulb temperature...')
    if fast:
        wb = wet_bulb_temperature_fast(temperature=temperature, dewpoint=dewpoint)
    else:
        wb = wet_bulb_temperature(pressure=pressure, temperature=temperature,
                                  dewpoint=dewpoint, vert_dim=vert_dim)
    

    mlh = freezing_level_height(temperature=wb, height=height, vert_dim=vert_dim)
    
    mlh.attrs['long_name'] = f'Melting-level height'
    mlh.attrs['description'] = f'Height of zero degree wet-bulb temperature isotherm.'
    mlh.name = 'melting_level'
    return mlh, wb

def isobar_temperature(pressure, temperature, isobar, vert_dim='model_level_number'):
    """
    Calculate the temperature at a given pressure.
    
    Arguments:
    
        - pressure: Pressure at each level [hPa].
        - temperature: Temperature at each level [K].
        - isobar: Pressure at which to find temperatures [hPa].
        - vert_dim: Name of vertical dimension.
        
    Returns:
    
        - Isobar temperature [K].
    """
    
    temp = log_interp(x=temperature, coords=pressure, at=isobar, dim=vert_dim)
    temp.attrs = {}
    temp.attrs['description'] = f'Temperature at {isobar} hPa.'
    temp.attrs['long_name'] = 'Isobar temperature'
    temp.attrs['units'] = 'K'
    return temp

def wind_shear(surface_wind_u, surface_wind_v, wind_u, wind_v, height, shear_height=6000, 
               vert_dim='model_level_number'):
    """
    Calculate wind shear.
    
    Arguments:
        - surface_wind_u, surface_wind_v: U and V components of surface wind.
        - wind_u, wind_v: U and V components of above-surface wind.
        - height: The height of every coordinate in wind_u and wind_v.
        - shear_height: The wind height to subtract from surface wind [m].
        - vert_dim: Name of vertical dimension.
        
    Returns:
    
        - Wind shear = wind at shear_height - wind at surface.
    """
    
    wind_high_u = linear_interp(x=wind_u, coords=height, at=shear_height, dim=vert_dim)
    wind_high_v = linear_interp(x=wind_v, coords=height, at=shear_height, dim=vert_dim)
    
    shear_u = wind_high_u - surface_wind_u
    shear_u.name = 'shear_u'
    shear_u.attrs['long_name'] = f'Surface to {shear_height} m wind shear, U component.'
    
    shear_v = wind_high_v - surface_wind_v
    shear_v.name = 'shear_v'
    shear_v.attrs['long_name'] = f'Surface to {shear_height} m wind shear, V component.'
    
    high_mag = np.sqrt(wind_high_u**2 + wind_high_v**2)
    surface_mag = np.sqrt(surface_wind_u**2 + surface_wind_v**2)
    positive_shear = high_mag > surface_mag
    positive_shear.name = 'positive_shear'
    positive_shear.attrs['long_name'] = f'True if {shear_height} wind > surface wind.'
    
    shear_magnitude = np.sqrt(shear_u**2 + shear_v**2)
    shear_magnitude.name = 'shear_magnitude'
    shear_magnitude.attrs['long_name'] = f'Surface to {shear_height} m bulk wind shear.'

    out = xarray.merge([shear_u, shear_v, shear_magnitude, positive_shear], 
                       combine_attrs='drop_conflicts')
    for v in ['shear_u', 'shear_v', 'shear_magnitude']:
        out[v].attrs['units'] = 'm s$^{-1}$'
        
    return out

def significant_hail_parameter(mucape, mixing_ratio, lapse, temp_500, shear, flh):
    """
    Calculate the significant hail parameter, as given at
    https://www.spc.noaa.gov/exper/mesoanalysis/help/help_sigh.html
    
    Arguments:
    
        - mucape: Most unstable parcel CAPE [J kg-1].
        - mixing_ratio: Mixing ratio of the most unstable parcel [kg kg-1].
        - lapse: 700-500 hPa lapse rate [K km-1].
        - temp_500: Temperature at 500 hPa [K].
        - shear: 0-6 km bulk wind shear [m s-1].
        - flh: Freezing level height [m].
        
    Returns:
    
        - Significant hail parameter.
    """

    # Convert from kg kg-1 to g kg-1
    mixing_ratio = mixing_ratio * 1e3

    # Use positive values of lapse rate.
    lapse = -lapse

    # Convert temperatures from K to C.
    temp_500 = temp_500 - 273.15

    # Apply thresholds on inputs to identify where SHIP is valid.
    shear = shear.where(shear >= 7).where(shear <= 27)
    mixing_ratio = mixing_ratio.where(mixing_ratio >= 11).where(mixing_ratio <= 13.6)
    temp_500 = temp_500.where(temp_500 <= -5.5, other=-5.5)

    # Calculate basic SHIP value.
    ship = mucape * mixing_ratio * lapse * -temp_500 * shear / 42000000 
    
    # Three conditions change the value of SHIP.
    ship = ship.where(mucape >= 1300, other=ship * (mucape/1300))
    ship = ship.where(lapse >= 5.8, other=ship * (lapse/5.8))
    ship = ship.where(flh >= 2400, other=ship * (flh/2400))
    
    # Metadata.
    ship.attrs['long_name'] = 'Significant hail parameter'
    ship.attrs['units'] = 'J kg$^{-2}$ g K$^2$ km$^{-1}$ m s$^{-1}$'
    
    return ship

def valid_data(dat, vert_dim):
    """
    Ensure that data is in the correct format to be passed to functions in this library. 
    
    Arguments:
        dat: The data to check.
        vert_dim: The name of the vertical dimension (e.g. level number).
    
    Returns: True if the data is correctly formated, False otherwise.
    """
    
    assert np.all(np.abs(dat[vert_dim].diff(dim=vert_dim)) == 1), 'Index increments must all be 1.'
    assert dat.pressure.diff(dim=vert_dim).max() < 0, 'Pressures must decrease with increasing level number.'
    return True
    
def storm_proxies(dat):
    """
    Calculate storm proxies.
    
    Arguments:
    
        - dat: Data as returned by conv_properties().
        
    Returns:
    
        - DataSet with proxy values (binary, 1=proxy triggered, 0=proxy untriggered).
    """
    
    # Ignore negative CAPE.
    dat = dat.rename({'shear_magnitude': 'S06'})
    dat['mixed_100_cape'] = dat.mixed_100_cape.where(dat.mixed_100_cape >= 0)
    dat['mixed_50_cape'] = dat.mixed_50_cape.where(dat.mixed_50_cape >= 0)
    dat['mu_cape'] = dat.mu_cape.where(dat.mu_cape >= 0)

    out = xarray.Dataset()

    # Proxy calculations.
    # Craven 2004.
    out['proxy_Craven2004'] = (dat.mixed_100_cape * dat.S06) >= 20000
    
    # Kunz 2007.
    out['proxy_Kunz2007'] = np.logical_or(dat.mixed_100_lifted_index <= -2.07,
                                          np.logical_or(dat.mu_cape >= 1474,
                                                        dat.mixed_100_dci >= 25.7))

    # Trapp 2007.
    out['proxy_Trapp2007'] = np.logical_and(dat.mixed_100_cape * dat.S06 >= 10000,
                                            dat.mixed_100_cape >= 100)
    out['proxy_Trapp2007'] = np.logical_and(out.proxy_Trapp2007, dat.S06 >= 5)
    out['proxy_Trapp2007'] = np.logical_and(out.proxy_Trapp2007, dat.positive_shear)

    # Marsh 2009.
    out['proxy_Marsh2009'] = (dat.mixed_100_cape * dat.S06) >= 10000
    
    # Allen 2011.
    out['proxy_Allen2011'] = dat.mixed_50_cape * dat.S06**1.67 >= 25000

    # Allen 2014.
    out['proxy_Allen2014'] = np.logical_and(out.proxy_Allen2011,
                                            dat.mixed_50_cin > -25)
    out['proxy_Allen2014'] = np.logical_and(out.proxy_Allen2014,
                                            dat.S06 > 7.5)
    out['proxy_Allen2014'] = np.logical_and(out.proxy_Allen2014,
                                            dat.lapse_rate_700_500 < -6.5)

    # Eccel 2012.
    out['proxy_Eccel2012'] = np.logical_and(dat.mixed_100_cape * dat.S06 > 10000, 
                                            dat.mixed_100_cin > -50)

    # Mohr 2013.
    out['proxy_Mohr2013'] = np.logical_or(dat.mixed_100_lifted_index <= -1.6,
                                          dat.mixed_100_cape >= 439)
    out['proxy_Mohr2013'] = np.logical_or(out.proxy_Mohr2013,
                                          dat.mixed_100_dci >= 26.4)

    # Significant hail parameter.
    out['ship'] = significant_hail_parameter(mucape=dat.mu_cape,
                                             mixing_ratio=dat.mu_mixing_ratio,
                                             lapse=dat.lapse_rate_700_500,
                                             temp_500=dat.temp_500,
                                             shear=dat.S06,
                                             flh=dat.freezing_level)
    out['proxy_SHIP_0.1'] = out.ship > 0.1
    out.ship.attrs['long_name'] = 'Significant hail parameter (SHIP)'

    # Define proxies and which study they are from.
    proxies = {'proxy_Craven2004': 'Craven 2004',
               'proxy_Kunz2007': 'Kunz 2007',
               'proxy_Trapp2007': 'Trapp 2007',
               'proxy_Marsh2009': 'Marsh 2009',
               'proxy_Allen2011': 'Allen 2011',
               'proxy_Allen2014': 'Allen 2014',
               'proxy_Eccel2012': 'Eccel 2012',
               'proxy_Mohr2013': 'Mohr 2013',
               'proxy_SHIP_0.1': 'SHIP > 0.1'}

    for proxy, val in proxies.items():
        out[proxy].attrs['long_name'] = 'Proxy ' + val
        
    return out

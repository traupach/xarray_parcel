# parcel_functions.py
#
# xarray-enabled versions of MetPy functions for atmospheric parcel calculations.
# 
# Author: Tim Raupach <t.raupach@unsw.edu.au>

import metpy
import xarray
import numpy as np
from metpy.units import units
import metpy.constants as mpconsts
        
def get_layer(dat, depth=100, drop=False, vert_dim='model_level_number', interpolate=True):
    """
    Return an atmospheric layer from the surface with a given depth.

    Arguments:
        dat: DataArray, must contain pressure.
        bottom: Pressure level to start from [hPa].
        depth: Depth above the bottom of the layer to mix [hPa].
        drop: Drop unselected elements?
        vert_dim: Vertical dimension name.
        interpolate: Interpolate the bottom/top layers?

    Returns: xarray DataArray with pressure and data variables for the layer.
    """
       
    # Use the surface (lowest level) pressure as the bottom pressure.
    bottom_pressure = dat.pressure.max(dim=vert_dim)
        
    # Calculate top pressure.
    if interpolate:
        top_pressure = bottom_pressure-depth
        interp_level = log_interp(x=dat, at=top_pressure, 
                                  coords=dat.pressure, dim=vert_dim)
        interp_level['pressure'] = top_pressure
        dat = insert_level(d=dat, level=interp_level, coords='pressure', vert_dim=vert_dim)
    else:
        top_pressure = bound_pressure(pressure=dat.pressure, 
                                      bound=bottom_pressure-depth, 
                                      vert_dim=vert_dim)
        
    # Select the layer.
    layer = dat.where(dat.pressure <= bottom_pressure, drop=drop)
    layer = dat.where(dat.pressure >= top_pressure, drop=drop)
    
    return layer

def most_unstable_parcel(dat, depth=300, drop=False, vert_dim='model_level_number'):
    """
    Return the most unstable parcel with an atmospheric layer from with the 
    requested bottom and depth. No interpolation is performed.

    Arguments:
        dat: DataArray, must contain pressure, temperature, and dewpoint.
        bottom: Pressure level to start from [hPa].
        depth: Depth above the bottom of the layer to mix [hPa].
        drop: Drop unselected elements?
        vert_dim: Vertical dimension name.

    Returns: xarray DataArray with pressure and data variables for the layer.
    """

    layer = get_layer(dat=dat, depth=depth, drop=drop, vert_dim=vert_dim, interpolate=False)
    eq = metpy.calc.equivalent_potential_temperature(pressure=layer.pressure,
                                                     temperature=layer.temperature,
                                                     dewpoint=layer.dewpoint).metpy.dequantify()
    max_eq = eq.max(dim=vert_dim)
    assert np.all(eq.where(eq == max_eq).count(vert_dim) == 1), 'Multiple maximum eq values.'
    most_unstable = layer.where(eq == max_eq).max(dim=vert_dim, keep_attrs=True)
    return most_unstable
    
def mixed_layer(dat, depth=100, vert_dim='model_level_number'):
    """
    Mix variable(s) over a layer, yielding a mass-weighted average.

    Integrate a data variable with respect to pressure and determine the
    average value using the mean value theorem.

    Arguments:
        dat: The DataArray to mix. Must contain pressure and variables.
        bottom: Pressure above the surface pressure to start from [hPa].
        depth: Depth above the bottom of the layer to mix [hPa].
        vert_dim: The name of the vertical dimension.

    Returns: xarray with mixed values of each data variable.
    """
    
    layer = get_layer(dat=dat, depth=depth, drop=True)
    
    pressure_depth = np.abs(layer.pressure.min(vert_dim) - 
                            layer.pressure.max(vert_dim))
   
    ret = (1. / pressure_depth) * trapz(dat=layer, x='pressure', dim=vert_dim)
    return ret
    
def trapz(dat, x, dim, mask=None):
    """ 
    Perform trapezoidal rule integration along an axis, ala numpy.trapz.
    Estimates int y dx.
   
    Arguments:
        dat: Data to process.
        x: The variable that contains 'x' values along dimension 'dim'.
        dim: The dimension along which to integrate 'y' values.
        mask: A mask the size of dx/means (ie dim.size-1) for which 
              areas to include in the integration.
    """

    dx = np.abs(dat[x].diff(dim))
    dx = dx.reset_coords(drop=True)
    means = dat.rolling({dim: 2}, center=True).mean(keep_attrs=True).dropna(dim, how='all')
    means = means.reset_coords(drop=True)

    dx = dx.assign_coords({dim: dx[dim]-1})
    means = means.assign_coords({dim: means[dim]-1})
    
    if mask is not None:
        dx = dx.where(mask)
        means = means.where(mask)
    
    return (dx * means).sum(dim)
    
def bound_pressure(pressure, bound, vert_dim='model_level_number'):
    """
    Calculate the bounding pressure in a layer; returns the closest pressure to the bound.
    
    Arguments:
        pressure: Atmospheric pressures [hPa].
        bound: Bound to retrieve, broadcastable to pressure [hPa].

    Returns: The bound pressures.
    """
    
    diffs = np.abs(pressure - bound)
    bounds = pressure.where(diffs == diffs.min(dim=vert_dim), drop=True)
    assert bounds[vert_dim].size == 1, 'Pressure field contains duplicates.'
    bounds = bounds.squeeze(drop=True)
    return bounds

def mixed_parcel(pressure, temperature, dewpoint, depth=100, vert_dim='model_level_number'):
    """
    Fully mix a layer of given depth above the surface and find the temparature, 
    pressure and dewpoint of the parcel.

    Arguments:
        pressure: Pressure by level [hPa].
        temperature: Temperature at each level [K].
        dewpoint: Dewpoint at each level [K].
        depth: Depth above the surface to mix [hPa].
        vert_dim: The name of the vertical dimension.

    Returns: DataArray with mixed parcel pressure [hPa], temperature [K] and dewpoint [K].
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
    mp = mixed_layer(xarray.merge([pressure, theta, mixing_ratio]), depth=depth)
        
    # Convert potential temperature back to temperature.
    mp['temperature'] = mp.theta * metpy.calc.exner_function(parcel_start_pressure)
    mp['temperature'] = mp.temperature.metpy.dequantify()
    mp.temperature.attrs['long_name'] = 'Mixed parcel temperature'
    mp.temperature.attrs['units'] = 'K'

    # Convert mixing ratio back to dewpoint.
    mp['vapour_pressure'] = metpy.calc.vapor_pressure(parcel_start_pressure, mp.mixing_ratio)
    mp['vapour_pressure'] = mp.vapour_pressure.metpy.dequantify()
    mp.vapour_pressure.attrs['long_name'] = 'Mixed-parcel vapour pressure'
    
    mp['dewpoint'] = metpy.calc.dewpoint(mp.vapour_pressure).metpy.convert_units('K')
    mp['dewpoint'] = mp.dewpoint.metpy.dequantify()
    mp.dewpoint.attrs['long_name'] = 'Mixed-parcel dewpoint'
    
    # For pressure, use the starting pressure for the layer (following MetPy's 
    # mixed_parcel function).
    mp['pressure'] = parcel_start_pressure 
    
    return mp

def dry_lapse(pressure, parcel_temperature, parcel_pressure=None, vert_dim='model_level_number'):
    """
    Calculate the temperature of a parcel raised dry-adiabatically (conserving
    potential temperature).

    Arguments:
        pressure: Atmospheric pressure level(s) of interest [hPa].
        parcel_temperature: Parcel temperature before lifting (constant or broadcast-able 
                            DataArray).
        parcel_pressure: Parcel pressure(s) before lifting. Defaults to vertical maximum.
        vert_dim: The name of the vertical dimension.

    Returns: Parcel temperature at each pressure level.
    """
    
    if parcel_pressure is None:
        parcel_pressure = pressure.max(vert_dim)
    out = parcel_temperature * (pressure / parcel_pressure)**mpconsts.kappa
    out.attrs['long_name'] = 'Dry lapse rate temperature'
    out.attrs['units'] = 'K'
    return out

def moist_adiabat_tables(regenerate=False, cache=True,
                         lookup_cache='lookup_tables/moist_adiabat_lookup.nc',
                         adiabats_cache='lookup_tables/adiabats_cache.nc',
                         **kwargs):
    """
    Calculate moist adiabat lookup tables.
    
    Arguments:
        regenerate: Calculate from scratch and save caches?
        cache: Write cache files?
        lookup_cache: A cache file (nc) for the adiabat lookup table.
        adiabats_cache: A cache file (nc) for the adiabats cache.
        **kwargs: Keyword arguments to moist_adiabat_lookup().
                           
    Returns: two DataArrays: 1) a lookup table of pressure/temperature vs. adiabat number, 
             and 2) a lookup table of adiabat number to temperature by pressure profiles.
    """
    
    if not regenerate:
        adiabat_lookup = xarray.open_dataset(lookup_cache)
        adiabats = xarray.open_dataset(adiabats_cache)
        return adiabat_lookup, adiabats
    
    # Generate lookup tables.
    adiabat_lookup, adiabats = moist_adiabat_lookup(**kwargs)
    
    if cache:
        adiabats.to_netcdf(adiabats_cache)
        adiabat_lookup.to_netcdf(lookup_cache)
        
    return adiabat_lookup, adiabats

def round_to(x, to, dp=2):
    """
    Round x to the nearest 'to' and return rounded to 'dp' decimal points.
    """
    return np.round(np.round(x / to) * to, dp)

def moist_adiabat_lookup(pressure_levels=np.round(np.arange(1100, 0, step=-0.5), 1),
                         temperatures=np.round(np.arange(173, 316, step=0.02), 2),
                         pres_step=0.5, temp_step=0.02):
    """
    Calculate moist adiabat lookup tables.
    
    Arguments:
        pressure_levels: Pressure levels to keep in adiabat lookup table [hPa].
        temperatures: Temperatures to keep in adiabat lookup table [K].
        pres_step, temp_step: (Positive) step size for pressure_levels and 
                              temperatures, respectively.
                              
    Returns: two DataArrays: 1) a lookup table of pressure/temperature vs. adiabat number, 
             and 2) a lookup table of adiabat number to temperature by pressure profiles.
    """
        
    curves = []
    adiabat_lookup = xarray.Dataset({'adiabat': np.nan})
    adiabat_lookup = adiabat_lookup.expand_dims({'pressure': pressure_levels, 
                                                 'temperature': temperatures}).copy(deep=True)
    
    # Find the adiabat for each starting temperature.
    i = 1
    for parcel_temperature in temperatures:
        for offset in [0, temp_step/2]:
            profile_temps = metpy.calc.moist_lapse(pressure=pressure_levels*units.hPa, 
                                                   temperature=(parcel_temperature+offset)*units.K).m

            nearest_temps = round_to(profile_temps, temp_step)
            idx = np.isin(nearest_temps, temperatures)
            temp_idx = xarray.DataArray(nearest_temps[idx], dims=['idx'])
            pres_idx = xarray.DataArray(pressure_levels[idx], dims=['idx'])
            adiabat_lookup.adiabat.loc[{'pressure':pres_idx, 'temperature': temp_idx}] = i

            # In profile_temps we have the adiabat temperature for every pressure level.
            # But some temperatures in the lookup table may be missing. Interpolate the 
            # pressures for each required temperature.
            pres_per_temp = np.interp(x=temperatures, xp=profile_temps[::-1], 
                                      fp=pressure_levels[::-1], right=np.nan, left=np.nan)

            pres_per_temp = round_to(pres_per_temp, pres_step)
            idx = np.isin(pres_per_temp, pressure_levels)
            pres_idx = xarray.DataArray(pres_per_temp[idx], dims=['idx'])
            temp_idx = xarray.DataArray(temperatures[idx], dims=['idx'])
            adiabat_lookup.adiabat.loc[{'pressure':pres_idx, 'temperature': temp_idx}] = i

            # curves contains the adiabats themselves.
            curves.append(xarray.Dataset({'temperature': (['pressure'], profile_temps)}, 
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

def moist_lapse(pressure, parcel_temperature, moist_adiabat_lookup, moist_adiabats,
                parcel_pressure=None, vert_dim='model_level_number'):
    """
    Return the temperature of parcels raised moist-adiabatically (assuming liquid saturation processes).
    What is returned are approximate pseudo-adiabatic moist lapse rates, found using a lookup table.

    Arguments:
        pressure: Atmospheric pressure(s) to lift the parcel to [hPa].
        parcel_temperature: Temperature(s) of parcels to lift [K].
        moist_adiabat_lookup, moist_adiabats: Adiabat lookup tables generated by moist_adiabat_tables().
        parcel_pressure: Parcel pressure before lifting. Defaults to vertical maximum.
        vert_dim: The name of the vertical dimension.

    Returns: Parcel temperature at each pressure level.
    """

    if parcel_pressure is None:
        parcel_pressure = pressure.max(vert_dim)
        
    # For each starting parcel, find the moist adiabat that intersects the parcel pressure 
    # and temperature.
    adiabat_idx = moist_adiabat_lookup.sel({'pressure': parcel_pressure, 
                                            'temperature': parcel_temperature}, method='nearest')
    adiabat_idx = adiabat_idx.adiabat.reset_coords(drop=True)
    adiabats = moist_adiabats.sel(adiabat=adiabat_idx)
    
    # Interpolate the adiabat to get the temperature at each requested pressure.
    out = adiabats.temperature.interp({'pressure': pressure}).reset_coords(drop=True)
    return out

def lcl(parcel_pressure, parcel_temperature, parcel_dewpoint):
    """
    Return the lifting condensation level for parcels.
    
    Arguments:
        parcel_pressure: Pressure of the parcel to lift [hPa].
        parcel_temperature: Parcel temperature [K].
        parfel_dewpoint: Parcel dewpoint [K].
    
    Returns: A Dataset with lcl_pressure and lcl_temperature.
    """
    
    press_lcl, temp_lcl = metpy.calc.lcl(pressure=parcel_pressure, 
                                         temperature=parcel_temperature, 
                                         dewpoint=parcel_dewpoint)
    out = xarray.Dataset({'lcl_pressure': (parcel_temperature.dims, press_lcl.m),
                          'lcl_temperature': (parcel_temperature.dims, temp_lcl.m)})
    
    out.lcl_pressure.attrs['long_name'] = 'Lifting condensation level pressure'
    out.lcl_pressure.attrs['units'] = 'hPa'
    out.lcl_temperature.attrs['long_name'] = 'Lifting condensation level temperature'
    out.lcl_temperature.attrs['units'] = 'K'
    
    return out

def parcel_profile(pressure, parcel_pressure, parcel_temperature, parcel_dewpoint, 
                   moist_adiabat_lookup, moist_adiabats):
    """
    Calculate temperatures of a lifted parcel.
    
    Arguments:
        pressure: Pressure levels to calculate on [hPa].
        parcel_pressure: Pressure of the parcel [hPa].
        parcel_temperature: Temperature of the parcel [K].
        parcel_dewpoint: Dewpoint of the parcel [K].
        moist_adiabat_lookup, moist_adiabats: Adiabat lookup tables generated by moist_adiabat_tables().
  
    Returns: Dataset with the temperature of the parcel lifted from parcel_pressure to 
             levels in pressures, plus the LCL pressure and temperature.
    """
       
    out = xarray.Dataset()
    out['pressure'] = pressure

    # Find the LCL for the selected parcel.
    out = xarray.merge([out, lcl(parcel_pressure=parcel_pressure, 
                                 parcel_temperature=parcel_temperature, 
                                 parcel_dewpoint=parcel_dewpoint)])

    # Parcels are raised along the dry adiabats from the starting point to the LCL.
    below_lcl = dry_lapse(pressure=pressure, 
                          parcel_temperature=parcel_temperature, 
                          parcel_pressure=parcel_pressure)

    # Above the LCL parcels follow the moist adiabats from the LCL temp/pressure.
    above_lcl = moist_lapse(pressure=pressure, 
                            parcel_temperature=out.lcl_temperature,
                            parcel_pressure=out.lcl_pressure,
                            moist_adiabat_lookup=moist_adiabat_lookup,
                            moist_adiabats=moist_adiabats)

    out['temperature'] = below_lcl.where(pressure >= out.lcl_pressure, other=above_lcl)
    out.temperature.attrs['long_name'] = 'Lifted parcel temperature'
    out.temperature.attrs['units'] = 'K'

    out = out.reset_coords(drop=True)
    return out

def parcel_profile_with_lcl(pressure, temperature, parcel_pressure, parcel_temperature, 
                            parcel_dewpoint, moist_adiabat_lookup, moist_adiabats,
                            vert_dim='model_level_number'):
    """
    Calculate temperatures of a lifted parcel, including at the lcl.
    
    Arguments:
        pressure: Pressure levels to calculate on [hPa].
        temperature: Temperature at each pressure level [K].
        parcel_pressure: Pressure of the parcel [hPa].
        parcel_temperature: Temperature of the parcel [K].
        parcel_dewpoint: Dewpoint of the parcel [K].
        moist_adiabat_lookup, moist_adiabats: Adiabat lookup tables generated by 
                                              moist_adiabat_tables().
        vert_dim: The name of the vertical dimension.
  
    Returns: Dataset with the temperature of the parcel lifted from parcel_pressure to 
             levels in pressures, including the LCL, plus the LCL pressure and temperature, and
             environmental temperature including at the LCL.
    """
    
    profile = parcel_profile(pressure=pressure, parcel_pressure=parcel_pressure, 
                             parcel_temperature=parcel_temperature,
                             parcel_dewpoint=parcel_dewpoint, 
                             moist_adiabat_lookup=moist_adiabat_lookup, 
                             moist_adiabats=moist_adiabats)
    return add_lcl_to_profile(profile=profile, vert_dim=vert_dim, temperature=temperature)

def add_lcl_to_profile(profile, vert_dim='model_level_number', temperature=None):
    """
    Add the LCL to a profile.
    
    Arguments:
        profile: Profile as returned from parcel_profile().
        vert_dim: The vertical dimension to add the LCL pressure/temp to.
        temperature: Environmental temperatures. If provided, interpolate environment 
                     temperature for the LCL and return as 'env_temperature'.
        
    Returns: A new profile object with LCL pressure and temperatures added. 
             Note the vertical coordinate in the new profile is reindexed.
    """
    
    level = xarray.Dataset({'pressure': profile.lcl_pressure,
                            'temperature': profile.lcl_temperature})
    out = insert_level(d=profile, level=level, coords='pressure', vert_dim=vert_dim)
    out['lcl_pressure'] = profile.lcl_pressure
    out['lcl_temperature'] = profile.lcl_temperature
    out.temperature.attrs['long_name'] = 'Temperature with LCL'
    out.pressure.attrs['long_name']  = 'Pressure with LCL'

    if not temperature is None:
        environment = xarray.Dataset({'temperature': temperature,
                                      'pressure': profile.pressure})
        temp_at_level = xarray.Dataset({'temperature': linear_interp(x=temperature, 
                                                                     coords=profile.pressure,
                                                                     at=level.pressure, 
                                                                     dim=vert_dim),
                                        'pressure': level.pressure})

        environment = insert_level(d=environment, level=temp_at_level, 
                                   coords='pressure', vert_dim=vert_dim)
        out['environment_temperature'] = environment.temperature
        out.environment_temperature.attrs['long_name'] = 'Environment temperature'
        out.environment_temperature.attrs['units'] = 'K'
    
    return out

def insert_level(d, level, coords, vert_dim='model_level_number', fill_value=-999):
    """
    Insert a new level into a vertically sorted dataset.
    
    Arguments:
        d: The data to work on.
        coords: The coordinate name in d.
        level: The new values to add; a single layer with values for 'coord' 
               and any other variables to add.
        vert_dim: The vertical dimension to add new level to.
        
    Returns: A new object with the new level added.
             Note the vertical coordinate in the new profile is reindexed.
    """
    
    # To conserve nans in the original dataset, replace them with fill_value in
    # the coordinate array.
    #assert not np.any(d[coords] == fill_value), 'dataset d contains fill_value.'
    d = d.where(np.logical_not(np.isnan(d[coords])), other=fill_value)
    
    below = d.where(d[coords] > level[coords])
    above = d.where(d[coords] < level[coords])
       
    # Above the new coordinate, shift the vertical coordinate indices up one.
    above = above.assign_coords({vert_dim: d[vert_dim] + 1})

    # Use broadcasting to fills regions below the new coordinate.
    out, _ = xarray.broadcast(below, above)

    # Fill regions above the new coordinate.
    above, _ = xarray.broadcast(above, out)
    out = above.where(np.isnan(out[coords]), other=out)
    
    # Any remaining nan values must now be the new level, so fill those regions.
    new, _ = xarray.broadcast(level, out)
  
    # Subset to keys from new only.
    out = out[list(new.keys())]
    out = new.where(np.isnan(out), other=out)
    
    # Replace fill_value with nans.
    out = out.where(out != fill_value, other=np.nan)
        
    return out

def find_intersections(x, a, b, dim, log_x=False):
    """
    Find intersections of two lines that share x coordinates.
    
    Arguments:
        x: The shared x coordinate values.
        a: y values for line 1.
        b: y values for line 2.
        dim: The dimension along which the coordinates are indexed.
        log_x: Use a logarithmic transform on x coordinates (e.g. for pressure coords)?
        
    Returns: a Dataset containing x, y coordinates for all intersections, 
             increasing intersections and decreasing intersections. Note duplicates 
             are not removed.
    """

    if log_x:
        x = np.log(x)

    # Find intersections. Non-zero points in diffs indicates an intersection.
    diffs = np.sign(a - b).diff(dim=dim)
    
    # Identify the points after each intersection.
    after_intersects = diffs.where(diffs == 0, other=1)
    
    # And the points just before each intersection.s
    before_intersects = xarray.concat([xarray.zeros_like(a.isel({dim: 0})), 
                                       after_intersects], dim=dim)
    before_intersects = before_intersects.shift({dim: -1}, fill_value=0)
    
    # The sign of the change for the intersect.
    sign_change = np.sign(a.where(after_intersects == 1) - b.where(after_intersects == 1))

    x0 = x.where(before_intersects == 1).shift({dim: 1})
    x1 = x.where(after_intersects == 1)
    a0 = a.where(before_intersects == 1).shift({dim: 1})
    a1 = a.where(after_intersects == 1)
    b0 = b.where(before_intersects == 1).shift({dim: 1})
    b1 = b.where(after_intersects == 1) 

    # Calculate the x-intersection. This comes from finding the equations of the two lines,
    # one through (x0, a0) and (x1, a1) and the other through (x0, b0) and (x1, b1),
    # finding their intersection, and reducing with a bunch of algebra.
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)

    # Calculate the y-intersection of the lines. Just plug the x above into the equation
    # for the line through the a points.
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

def lfc_el(profile, vert_dim='model_level_number'):
    """
    Calculate the level of free convection (LFC) and equilibrium level (EL). 
    
    Works by finding the first intersection of the ideal parcel path and the measured parcel 
    temperature. If this intersection occurs below the LCL, the LFC is determined to be the 
    same as the LCL, based upon the conditions set forth in [USAF1990]_, pg 4-14, where a 
    parcel must be lifted dry adiabatically to saturation before it can freely rise.
    The LFC returned is the 'bottom' LFC with highest pressure; the EL returned is the 
    'top' EL with the lowest pressure.

    Arguments:
        profile: The parcel profile, including the LCL, as returned from 
                 parcel_profile_with_lcl().
        vert_dim: Vertical dimension name in input arrays.
    
    Returns: a DataArray with LFC pressure (lfc_pressure) and temperature (lfc_temperature).
    """
    
    # Find all intersections between parcel and environmental temperatures by pressure.
    intersections = find_intersections(x=profile.pressure, a=profile.temperature, 
                                       b=profile.environment_temperature, 
                                       dim=vert_dim, log_x=True)

    # Find intersections again, ignoring first level.
    intersections_above = find_intersections(x=profile.pressure.isel({vert_dim: slice(1,None)}),
                                             a=profile.temperature.isel({vert_dim: slice(1,None)}), 
                                             b=profile.environment_temperature.isel({vert_dim: slice(1,None)}), 
                                             dim=vert_dim, log_x=True).reindex_like(intersections)
    
    # For points for which the atmosphere and parcel temperatures have the same lowest-level value, 
    # ignore this point and find the real LFC above it.
    intersections = intersections.where((profile.environment_temperature.isel({vert_dim: 0}) != 
                                         profile.temperature.isel({vert_dim: 0})),
                                        other=intersections_above)

    out = xarray.Dataset()
    
    # By default the first values are the lowest (highest pressure) crossings for LFC and 
    # the highest (lowest pressure) crossings for EL. The LFC also has to be above the LCL.
    above_lcl = intersections.increasing_x < profile.lcl_pressure
        
    out['lfc_pressure'] = intersections.increasing_x.where(above_lcl).max(dim='offset_dim')
    out['lfc_temperature'] = intersections.increasing_y.where(intersections.increasing_x == 
                                                              out.lfc_pressure).max(dim='offset_dim')
    
    # Determine equilibrium pressure and temperature. The 'top' (lowest pressure) EL is returned.
    out['el_pressure'] = intersections_above.decreasing_x.min(dim='offset_dim')
    out['el_temperature'] = intersections_above.decreasing_y.where(intersections.decreasing_x == 
                                                                   out.el_pressure).max(dim='offset_dim')
    
    # If at the top of the atmosphere the parcel profile is warmer than the environment,
    # no EL exists. Also if EL is lower than or equal to LCL, no EL exists.
    top_pressure = profile.pressure == profile.pressure.min(dim=vert_dim)
    top_prof_temp = profile.temperature.where(top_pressure).max(dim=vert_dim)
    top_env_temp = profile.environment_temperature.where(top_pressure).max(dim=vert_dim)
    
    top_colder = top_prof_temp <= top_env_temp
    el_above_lcl = out.el_pressure < profile.lcl_pressure
    el_exists = np.logical_and(top_colder, el_above_lcl)
    out['el_pressure'] = out.el_pressure.where(el_exists, other=np.nan)
    out['el_temperature'] = out.el_temperature.where(el_exists, other=np.nan)
    
    # There should only be one LFC and EL per point.
    assert not 'offset_dim' in out.keys(), 'Error, duplicate crossings detected.'

    # Identify points where no LFC intersections were found.
    lfc_missing = np.isnan(intersections.increasing_x.max(dim='offset_dim'))

    # If no intersection was found, but a parcel temperature above the LCL is greater than the
    # environmental temperature, return the LCL.
    above_lcl = profile.pressure < profile.lcl_pressure
    pos_parcel = ((profile.temperature.where(above_lcl) > 
                   profile.environment_temperature.where(above_lcl)).any(dim=vert_dim))
    no_lfc_pos_parcel = np.logical_and(pos_parcel, lfc_missing)

    # Also return LCL if an intersection exists but all intersections are below the LCL
    # and EL is above the LCL.
    exists_but_na = np.logical_and(np.logical_not(lfc_missing), np.isnan(out.lfc_pressure))
    el_above_lcl = out.el_pressure < profile.lcl_pressure
    lfc_below_el_above = np.logical_and(exists_but_na, el_above_lcl)
    
    # Do the replacements with LCL.
    replace_with_lcl = np.logical_or(no_lfc_pos_parcel, lfc_below_el_above)
    out['lfc_pressure'] = profile.lcl_pressure.where(replace_with_lcl, other=out.lfc_pressure)
    out['lfc_temperature'] = profile.lcl_temperature.where(replace_with_lcl, other=out.lfc_temperature)   
    
    # Assign metadata.
    out.el_pressure.attrs['long_name'] = 'Equilibrium level pressure'
    out.el_pressure.attrs['units'] = 'hPa'
    out.el_temperature.attrs['long_name'] = 'Equilibrium level temperature'
    out.el_temperature.attrs['units'] = 'K'
    out.lfc_pressure.attrs['long_name'] = 'Level of free convection pressure'
    out.lfc_pressure.attrs['units'] = 'hPa'
    out.lfc_temperature.attrs['long_name'] = 'Level of free convection temperature'
    out.lfc_temperature.attrs['units'] = 'K'

    return out

def trap_around_zeros(x, y, dim, log_x=True, start=0):
    """
    Calculate dx * y for points just before and after zeros in y.
    
    Arguments:
        x: arrays of x along dim.
        y: arrays of y along dim.
        dim: Dimension along which to calculate.
        log_x: Log transform x?
        start: Zero-based position along dim to look for zeros.
        
    Returns: a Dataset containin the areas and x coordinates for each rectangular area 
    calculated before and after each zero; and an array of x coordinates that should be 
    replaced by the new areas if integrating along x and including these areas afterwards.
    """
    
    # Estimate zero crossings.
    zeros = xarray.zeros_like(y)
    zero_intersections = find_intersections(x=x.isel({dim: slice(start, None)}), 
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
    before_zeros_mask = xarray.concat([zero_level, zero_y], dim=dim).shift({dim: -1})
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
        areas = areas.dropna(dim=dim, how='all').reset_coords(drop=True)
        
        return(areas)
        
    areas_before_zeros = calc_areas(x=x, y=y, mask=before_zeros_mask, shift_x=1)
    areas_after_zeros = calc_areas(x=x, y=y, mask=after_zeros_mask, shift_x=0)
   
    # Concatenate areas before zeros and areas after zeros.
    areas = xarray.concat([areas_before_zeros, areas_after_zeros], dim=dim)
    #areas = areas.assign_coords({dim: np.arange(0, len(areas[dim]))})
    
    # Determine start/end points on x axis for each area.
    areas['x_from'] = areas.x - areas.dx/2
    areas['x_to'] = areas.x + areas.dx/2
    
    # Mask is a mask that selects elements that were *not* included in the differences;
    # to be used by a CAPE calculation where we don't want to count the areas around
    # zeros twice.
    mask = xarray.full_like(x, True)
    mask, bef = xarray.broadcast(mask, areas_before_zeros)
    mask = mask.where(np.isnan(bef.area), other=False)
    
    return areas, mask
   
def cape_cin_base(pressure, temperature, lfc_pressure, el_pressure, parcel_profile, 
                  vert_dim='model_level_number'):
    """
    Calculate CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile and parcel path. CIN is integrated between the surface and
    LFC, CAPE is integrated between the LFC and EL (or top of sounding). Intersection points
    of the measured temperature profile and parcel profile are logarithmically interpolated.
    
    Uses the bottom (highest-pressure) LFC and the top (lowest-pressure) EL.

    Formula adopted from [Hobbs1977]_.

    .. math:: \text{CAPE} = -R_d \int_{LFC}^{EL} (T_{parcel} - T_{env}) d\text{ln}(p)
    .. math:: \text{CIN} = -R_d \int_{SFC}^{LFC} (T_{parcel} - T_{env}) d\text{ln}(p)
    
    * :math:`CAPE` is convective available potential energy
    * :math:`CIN` is convective inhibition
    * :math:`LFC` is pressure of the level of free convection
    * :math:`EL` is pressure of the equilibrium level
    * :math:`SFC` is the level of the surface or beginning of parcel path
    * :math:`R_d` is the gas constant
    * :math:`g` is gravitational acceleration
    * :math:`T_{parcel}` is the parcel temperature
    * :math:`T_{env}` is environment temperature
    * :math:`p` is atmospheric pressure

    Arguments:
        pressure: Pressure level(s) of interest [hPa].
        temperature: Temperature at each pressure level [K].
        lfc_pressure: Pressure of level of free convection [hPa].
        el_pressure: Pressure of equilibrium level [hPa].
        parcel_profile: The parcel profile as returned from parcel_profile().
        vert_dim: The vertical dimension.

    Returns: Dataset with convective available potential energy (cape) and 
             convective inhibition (cin), both in J kg-1.
    """

    # Where the EL is nan, use the highest (lowest-pressure) value as the EL.
    el_pressure = pressure.min(dim=vert_dim).where(np.isnan(el_pressure), 
                                                   other=el_pressure)

    # Difference between the parcel path and measured temperature profiles.
    temp_diffs = xarray.Dataset({'temp_diff': (parcel_profile.temperature - temperature),
                                 'pressure': pressure,
                                 'log_pressure': np.log(pressure)})
    
    # Integration areas around zero differences. Note MetPy's implementation 
    # in _find_append_zero_crossings() looks for intersections from the 2nd 
    # index onward (start=1 in this code); but in this implemnetation the 
    # whole array needs to be checked (start=0) for the unit tests to pass.
    areas_around_zeros, trapz_mask = trap_around_zeros(x=temp_diffs.pressure, 
                                                       y=temp_diffs.temp_diff,  
                                                       dim=vert_dim, log_x=True)
    areas_around_zeros['x'] = np.exp(areas_around_zeros.x)
    areas_around_zeros['x_from'] = np.exp(areas_around_zeros.x_from)
    areas_around_zeros['x_to'] = np.exp(areas_around_zeros.x_to)
     
    # Integrate between LFC and EL pressure levels to get CAPE.
    diffs_lfc_to_el = temp_diffs.where(pressure <= lfc_pressure)
    diffs_lfc_to_el = diffs_lfc_to_el.where(pressure >= el_pressure)
    areas_lfc_to_el = areas_around_zeros.where(areas_around_zeros.x <= lfc_pressure)
    areas_lfc_to_el = areas_lfc_to_el.where(areas_around_zeros.x >= el_pressure)
    
    cape = mpconsts.Rd.m * trapz(dat=diffs_lfc_to_el, x='log_pressure', 
                                 dim=vert_dim, mask=trapz_mask)
    cape = cape.reset_coords().temp_diff
    cape = cape + (mpconsts.Rd.m * areas_lfc_to_el.area.sum(dim=vert_dim))
    cape.name = 'cape'
    cape.attrs['long_name'] = 'Convective available potential energy'
    cape.attrs['units'] = 'J kg-1'

    # Integrate between surface and LFC to get CIN.
    temp_diffs_surf_to_lfc = temp_diffs.where(pressure >= lfc_pressure)
    areas_surf_to_lfc = areas_around_zeros.where(areas_around_zeros.x >= lfc_pressure)
    cin = mpconsts.Rd.m * trapz(dat=temp_diffs_surf_to_lfc, x='log_pressure', 
                                dim=vert_dim, mask=trapz_mask)
    cin = cin.reset_coords().temp_diff
    cin = cin + (mpconsts.Rd.m * areas_surf_to_lfc.area.sum(dim=vert_dim))
    cin.name = 'cin'
    cin.attrs['long_name'] = 'Convective inhibition'
    cin.attrs['units'] = 'J kg-1'

    # Set any positive values for CIN to 0.
    cin = cin.where(cin <= 0, other=0)
    
    #return(areas_surf_to_lfc)
    return xarray.merge([cape, cin])

def cape_cin(pressure, temperature, parcel_temperature, parcel_pressure, parcel_dewpoint,
             moist_adiabat_lookup, moist_adiabats, vert_dim='model_level_number', 
             return_profile=False):
    """
    Calculate CAPE and CIN; wraps finding of LFC and parcel profile and call to 
    cape_cin_base. Uses the bottom (highest-pressure) LFC and the top (lowest-pressure) EL.

    Arguments:
        pressure: Pressure level(s) of interest [hPa].
        temperature: Temperature at each pressure level [K].
        parcel_temperature: The temperature of the starting parcel [K].
        parcel_pressure: The pressure of the starting parcel [K].
        parcel_dewpoint: The dewpoint of the starting parcel [K].
        moist_adiabat_lookup, moist_adiabats: Adiabat lookup tables generated by 
                                              moist_adiabat_tables().
        vert_dim: The vertical dimension.
        return_profile: Also return the lifted profile?

    Returns: Dataset with convective available potential energy (cape) and 
             convective inhibition (cin), both in J kg-1, plus the lifted profile if 
             return_profile is True.
    """
    
    # Calculate parcel profile.
    profile = parcel_profile_with_lcl(pressure=pressure,
                                      temperature=temperature,
                                      parcel_temperature=parcel_temperature,
                                      parcel_pressure=parcel_pressure,
                                      parcel_dewpoint=parcel_dewpoint,
                                      moist_adiabat_lookup=moist_adiabat_lookup, 
                                      moist_adiabats=moist_adiabats, 
                                      vert_dim=vert_dim)
    
    # Calculate LFC and EL.
    parcel_lfc_el = lfc_el(profile=profile)
    
    # Calculate CAPE and CIN.
    cape_cin = cape_cin_base(pressure=profile.pressure,
                             temperature=profile.environment_temperature, 
                             lfc_pressure=parcel_lfc_el.lfc_pressure, 
                             el_pressure=parcel_lfc_el.el_pressure, 
                             parcel_profile=profile)
    
    if return_profile:
        return cape_cin, profile
    else:
        return cape_cin
    
def surface_based_cape_cin(pressure, temperature, dewpoint, moist_adiabat_lookup, 
                           moist_adiabats, vert_dim='model_level_number', return_profile=False):
    """
    Calculate surface-based CAPE and CIN.

    Arguments:
        pressure: Pressure level(s) of interest [hPa].
        temperature: Temperature at each pressure level [K].
        dewpoint: Dewpoint at each level [K].
        moist_adiabat_lookup, moist_adiabats: Adiabat lookup tables generated by 
                                              moist_adiabat_tables().
        vert_dim: The vertical dimension.
        return_profile: Also return the lifted profile?
        
    Returns: Dataset with convective available potential energy (cape) and 
             convective inhibition (cin), both in J kg-1.
    """
    
    # Profile for surface-based parcel ascent.
    return cape_cin(pressure=pressure,
                    temperature=temperature,
                    parcel_temperature=temperature.isel({vert_dim: 0}),
                    parcel_pressure=pressure.isel({vert_dim: 0}),
                    parcel_dewpoint=dewpoint.isel({vert_dim: 0}),
                    moist_adiabat_lookup=moist_adiabat_lookup, 
                    moist_adiabats=moist_adiabats,
                    vert_dim=vert_dim,
                    return_profile=return_profile)

def most_unstable_cape_cin(pressure, temperature, dewpoint, moist_adiabat_lookup, moist_adiabats, 
                           vert_dim='model_level_number', depth=300, return_profile=False):
    """
    Calculate CAPE and CIN for the most unstable parcel within a given 
    depth above the surface..

    Arguments:
        pressure: Pressure level(s) of interest [hPa].
        temperature: Temperature at each pressure level [K].
        moist_adiabat_lookup, moist_adiabats: Adiabat lookup tables generated by 
                                              moist_adiabat_tables().
        dewpoint: Dewpoint at each level [K].
        vert_dim: The vertical dimension.
        depth: The depth above the surface (lowest-level pressure) in which to 
               look for the most unstable parcel.
        return_profile: Also return the lifted profile?
        
    Returns: Dataset with convective available potential energy (cape) and 
             convective inhibition (cin), both in J kg-1.
    """
    
    assert pressure.name == 'pressure', 'Pressure requires name pressure.'
    assert temperature.name == 'temperature', 'Temperature requires name temperature.'
    assert dewpoint.name == 'dewpoint', 'Dewpoint requires name dewpoint.'
        
    # Find the most unstable layer in the lowest 'depth' hPa.
    unstable_layer = most_unstable_parcel(dat=xarray.merge([pressure, temperature, dewpoint]), 
                                          depth=depth, vert_dim=vert_dim)
        
    # Subset to layers at or above the most unstable parcels.
    above_unstable = pressure <= unstable_layer.pressure
    pressure = shift_out_nans(pressure.where(above_unstable, drop=True), dim=vert_dim)
    temperature = shift_out_nans(temperature.where(above_unstable, drop=True), dim=vert_dim)
        
    return cape_cin(pressure=pressure,
                    temperature=temperature,
                    parcel_temperature=unstable_layer.temperature, 
                    parcel_pressure=unstable_layer.pressure,
                    parcel_dewpoint=unstable_layer.dewpoint,
                    moist_adiabat_lookup=moist_adiabat_lookup, 
                    moist_adiabats=moist_adiabats,
                    vert_dim=vert_dim,
                    return_profile=return_profile)    
        
def mixed_layer_cape_cin(pressure, temperature, dewpoint, moist_adiabat_lookup, moist_adiabats, 
                         vert_dim='model_level_number', depth=100, return_profile=False):
    """
    Calculate CAPE and CIN for a fully-mixed lowest x hPa parcel.

    Arguments:
        pressure: Pressure level(s) of interest [hPa].
        temperature: Temperature at each pressure level [K].
        moist_adiabat_lookup, moist_adiabats: Adiabat lookup tables generated by 
                                              moist_adiabat_tables().
        dewpoint: Dewpoint at each level [K].
        vert_dim: The vertical dimension.
        depth: The depth above the surface (lowest-level pressure) to mix [hPa].
        return_profile: Also return the lifted profile?
        
    Returns: Dataset with convective available potential energy (cape) and 
             convective inhibition (cin), both in J kg-1.
    """

    # Mix the lowest x hPa.
    mp = mixed_parcel(pressure=pressure, temperature=temperature, 
                      dewpoint=dewpoint, depth=depth)
    
    # Remove layers that were part of the mixed layer.
    higher_levels = pressure < (pressure.max(dim=vert_dim) - depth)
    pressure = shift_out_nans(pressure.where(higher_levels, drop=True), dim=vert_dim)
    temperature = shift_out_nans(temperature.where(higher_levels, drop=True), dim=vert_dim)
    
    # Add the mixed layer to the bottom of the profiles.
    mp[vert_dim] = pressure[vert_dim].min() - 1
    pressure = xarray.concat([mp.pressure, pressure], dim=vert_dim)
    temperature = xarray.concat([mp.temperature, temperature], dim=vert_dim)
        
    return cape_cin(pressure=pressure,
                    temperature=temperature,
                    parcel_temperature=mp.temperature, 
                    parcel_pressure=mp.pressure,
                    parcel_dewpoint=mp.dewpoint,
                    moist_adiabat_lookup=moist_adiabat_lookup, 
                    moist_adiabats=moist_adiabats, 
                    vert_dim=vert_dim,
                    return_profile=return_profile)
        
def shift_out_nans(x, dim, pt=0):
    """
    Shift data along a dim to remove all leading nans in that dimension, element-wise.
    
    Arguments:
        x: The data to work on.
        dim: The dimension to shift.
        pt: The point along the dimension to shift 'to'.
    """
    
    while np.any(np.isnan(x.isel({dim: pt}))):
        shifted = x.shift({dim: -1})
        x = shifted.where(np.isnan(x.isel({dim: pt})), other=x)
        
    return x

def lifted_index(profile, vert_dim='model_level_number'):
    """
    Calculate the lifted index. 
    
    Lifted index formula derived from [Galway1956]_ and referenced by [DoswellSchultz2006]_:
    
    .. math:: \text{LI} = T500 - Tp500
    
    * :math:`T500` is environmental temperature at 500 hPa.
    * :math:`Tp500` is temperature of lifted parcel at 500 hPa.

    Arguments:
        profile: Profile as returned by parcel_profile_with_lcl().
        vert_dim: The vertical dimension name.

    Returns: Lifted index at each point [K].
    """
    
    # Interpolate (linearly) to get 500 hPa values.
    dat = linear_interp(x=profile, coords=profile.pressure, at=500, dim=vert_dim)
    dat = dat.reset_coords(drop=True)
    
    # Calculate lifted index.
    li = xarray.Dataset({'lifted_index': dat.environment_temperature - dat.temperature})
    li.lifted_index.attrs['long_name'] = 'Lifted index'
    li.lifted_index.attrs['units'] = 'K'
    
    return li
    
def linear_interp(x, coords, at, dim='model_level_number'):
    """
    Perform simple linear interpolation to get values at specified points.
    
    Arguments:
        x: Data set to interpolate.
        coords: Coordinate value for each point in x.
        at: Points at which to interpolate.
        dim: The dimension along which to interpolate.
    
    It is assumed that x[coords] is sorted and does not contain duplicate 
    values along the selected dimension.
    """
    
    coords_before = coords.where(coords >= at).min(dim=dim)
    coords_after = coords.where(coords <= at).max(dim=dim)
    assert dim not in coords_before.coords, 'Duplicates detected in coords.'
    assert dim not in coords_after.coords, 'Duplicates detected in coords.'

    x_before = x.where(coords == coords_before).max(dim=dim)
    x_after = x.where(coords == coords_after).max(dim=dim)

    # The interpolated values.
    res = x_before + (x_after - x_before) * ((at - coords_before) / (coords_after - coords_before))
    
    # When the interpolated x exists already, return it.
    res = x_before.where(x_before == x_after, other=res)
    
    return(res)

def log_interp(x, coords, at, dim='model_level_number'):
    """
    Run linear_interp on logged coordinate values.
    
    Arguments:
        x: Data set to interpolate.
        coords: Coordinate value for each point in x.
        at: Points at which to interpolate.
        dim: The dimension along which to interpolate.
    
    It is assumed that x[coords] is sorted and does not contain duplicate 
    values along the selected dimension.
    """
    
    return linear_interp(x=x, coords=np.log(coords), at=np.log(at), dim=dim)
    
def deep_convective_index(pressure, temperature, dewpoint, lifted_index, 
                          vert_dim='model_level_number'):
    """
    Calculate the deep convective index (DCI) as defined by Kunz 2009.
    
    Arguments:
        pressure: Pressure values [hPa].
        temperature: Temperature at each pressure [K].
        dewpoint: Dewpoint temperature at each pressure.
        lifted_index: The lifted index.
        vert_dim: The vertical dimension name.
    
    Returns: the DCI [C] per point.
    """
    
    dat = xarray.merge([pressure, temperature, dewpoint])
    
    # Interpolate (linearly) to get 850 hPa values.
    dat = linear_interp(x=dat, coords=dat.pressure, at=850, dim=vert_dim)
    dat = dat.reset_coords(drop=True)
    
    # Convert temperature and dewpoint from K to C.
    dat['temperature'] = dat.temperature - 273.15
    dat['dewpoint'] = dat.dewpoint - 273.15
    
    dci = xarray.Dataset({'dci': dat.temperature + dat.dewpoint - lifted_index})
    dci.dci.attrs['long_name'] = 'Deep convective index'
    dci.dci.attrs['units'] = 'C'
    return dci
 
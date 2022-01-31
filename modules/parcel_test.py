# parcel_test.py
#
# Routines to test xarray-enabled versions of MetPy functions for atmospheric parcel 
# calculations.
#
# Author: Tim Raupach <t.raupach@unsw.edu.au>

import time
import metpy
import xarray
import numpy as np
from metpy.units import units
from metpy.units import concatenate
import modules.parcel_functions as parcel

# Load xarray-parcel's lookup tables.
parcel.load_moist_adiabat_lookups()

def time_function(func, dat, **kwargs):
    """
    Run a function and return its processing time.
    
    Arguments:
        func: The function to run.
        dat: An argument to pass to func.
        kwargs: Extra arguments to function?
        
    Returns: The processing time [s].
    """
    
    start = time.perf_counter()
    ret = func(dat, **kwargs)
    end = time.perf_counter()
    return ret, end-start

def compare(x, y, name, tolerance=1e-5):
    """
    Compare DataArray x to y.
    
    Arguments:
        x, y: DataArrays to compare with y as the reference.
        name: The name of the comparison to report.
    """
    
    x, y = xarray.broadcast(x, y)
    
    diffs = np.abs(x - y)
    reldiffs = diffs / y * 100
    max_rel_diff = np.round(np.max(diffs/y*100).values, 2)
    max_diff = np.round(np.max(diffs).values, 5)
    
    comp = max_diff < tolerance
    if not comp:
        name_and_unit = x.attrs['long_name'] + ' [' + x.attrs['units'] + '] (' + name + ')'
        max_diff = str(max_diff) + ' ' + x.attrs["units"]
        max_rel_diff = str(max_rel_diff) + '%'
        print(f'{name_and_unit:65} {max_diff:20} {max_rel_diff:20}')
        
    if not np.isnan(x).equals(np.isnan(y)):
        print(f'NaNs differ in {name}')
    
    return comp
    
def lcl_serial(parcel_pressure, parcel_temperature, parcel_dewpoint, 
               x_dim='longitude', y_dim='latitude'):
    """
    A looped wrapper for the metpy lcl function, for testing.
    DataArrays are assumed to contain latitude/longitude coordinates.
    
    Arguments:
        parcel_pressure: Parcel pressure [hPa].
        parcel_temperature: Parcel temperature [K].
        parcel_dewpoint: Parcel dewpoint [K].
        x_dim, y_dim: Names of x and y coordinate dimensions.
        
    Returns: An array of levels of free convection pressure and temperatures.
    """
    
    out = []
    for x in parcel_pressure[x_dim].values:
        for y in parcel_pressure[y_dim].values:
            
            pres = float(parcel_pressure.sel({x_dim: x, y_dim: y}).values) * units.hPa
            temp = float(parcel_temperature.sel({x_dim: x, y_dim: y}).values) * units.K
            dewpoint = float(parcel_dewpoint.sel({x_dim: x, y_dim: y}).values) * units.K
            
            press_lcl, temp_lcl = metpy.calc.lcl(pressure=pres, temperature=temp, dewpoint=dewpoint)
            
            lcl = xarray.Dataset({'lcl_pressure': press_lcl.m, 
                                  'lcl_temperature': temp_lcl.m,
                                  'lcl_virtual_temperature': np.nan})
            
            lcl = lcl.expand_dims({x_dim: [x]})
            lcl = lcl.expand_dims({y_dim: [y]})
            out.append(lcl)
            
    out = xarray.merge(out)
    
    out.lcl_pressure.attrs['long_name'] = 'Lifting condensation level pressure'
    out.lcl_pressure.attrs['units'] = 'hPa'
    out.lcl_temperature.attrs['long_name'] = 'Lifting condensation level temperature'
    out.lcl_temperature.attrs['units'] = 'K'
    
    return out
    
def moist_lapse_serial(pressure, parcel_temperature,
                       parcel_pressure=None, x_dim='longitude', y_dim='latitude', 
                       vert_dim='model_level_number'):
    """
    A vectorised wrapper for the metpy serial implementation of moist_lapse, for testing
    purposes. DataArrays are assumed to contain latitude/longitude coordinates.

    Arguments:
        pressure: Atmospheric pressure(s) to lift the parcel to [hPa].
        parcel_temperature: Temperature(s) of parcels to lift [K].
        parcel_pressure: Parcel pressure before lifting. 
        x_dim, y_dim: Names of x and y coordinate dimensions.
        vert_dim: The name of the vertical dimension.

    Returns: Parcel temperature at each pressure level.
    """
    
    if parcel_pressure is None:
        parcel_pressure = pressure.max(vert_dim)
    
    out = []
    for x in pressure[x_dim].values:
        for y in pressure[y_dim].values:
            
            pres = pressure.sel({x_dim: x, y_dim: y})
            temp = parcel_temperature.sel({x_dim: x, y_dim: y})
            if isinstance(parcel_pressure, int):
                ref_pressure = xarray.DataArray(parcel_pressure)
                ref_pressure.attrs['units'] = 'hPa'
            else:
                ref_pressure = parcel_pressure.sel({x_dim: x, y_dim: y})
            
            moist =  moist_lapse_single_point(pressure=pres, parcel_temperature=temp, 
                                              parcel_pressure=ref_pressure)
            
            out.append(xarray.Dataset(data_vars={'temperature': (['model_level_number'], moist.values)},
                                      coords={vert_dim: pressure[vert_dim], x_dim: x, y_dim: y}))
            
            out[-1] = out[-1].expand_dims({x_dim: [out[-1][x_dim]]})
            out[-1] = out[-1].expand_dims({y_dim: [out[-1][y_dim]]})
            out[-1] = out[-1].reset_coords(drop=True)
           
    out = xarray.merge(out)
    out.temperature.attrs['long_name'] = 'Moist lapse rate temperature'
    out.temperature.attrs['units'] = 'K'
    return out.temperature
    
def moist_lapse_single_point(pressure, parcel_temperature, parcel_pressure=None, 
                             vert_dim='model_level_number'):
    """
    A wrapper for the metpy implementation of moist_lapse, for testing purposes, for a single point.

    Arguments:
        pressure: Atmospheric pressure(s) to lift the parcel to [hPa].
        parcel_temperature: Temperature(s) of parcels to lift [K].
        parcel_pressure: Parcel pressure before lifting. 
        vert_dim: The name of the vertical dimension.

    Returns: Parcel temperature at each pressure level.
    """
    
    idx = np.logical_not(np.isnan(pressure.values))
    pres = pressure.values[idx] * units(pressure.attrs['units'])
    temp = parcel_temperature.values * units(parcel_temperature.attrs['units'])
    ref_pressure = parcel_pressure.values * units(parcel_pressure.attrs['units']) 
    
    moist = metpy.calc.moist_lapse(pressure=pres, temperature=temp, 
                                   reference_pressure=ref_pressure)
    
    # Assume that nans will only appear at the top (end) of pressure arrays.
    moist = np.concatenate([moist.m, pressure.values[np.isnan(pressure.values)]])
    
    out = xarray.Dataset({'temperature': (vert_dim, moist)},
                           coords={vert_dim: pressure[vert_dim]})
    out = out.reset_coords(drop=True).to_array().squeeze()
    out.name = 'temperature'
    return out

def surface_cape_serial(dat):
    """
    Loop over points and calculate surface CAPE and CIN for each point.
    
    Arguments:
        dat: An xarray Dataset containing dewpoint, pressure, temperature, and 
             specific humidity.
            
    Returns: Dataset containing CAPE and CIN values.
    """
    
    out = []
    
    for lat in dat.latitude.values:
        for lon in dat.longitude.values:
            # Assign units to values.
            pres = dat.sel(latitude=lat, longitude=lon).pressure.values * units(dat.pressure.attrs['units'])
            temp = dat.sel(latitude=lat, longitude=lon).temperature.values * units(dat.temperature.attrs['units'])
            spec_hum = (dat.sel(latitude=lat, longitude=lon).specific_humidity.values *
                        units(dat.specific_humidity.attrs['units']))
    
            # Dewpoint.
            dewpoint = metpy.calc.dewpoint_from_specific_humidity(pressure=pres,
                                                                  temperature=temp,
                                                                  specific_humidity=spec_hum).to('K')
    
            # Surface-based CAPE and CIN.
            surface_cape, surface_cin = metpy.calc.surface_based_cape_cin(pressure=pres,
                                                                          temperature=temp,
                                                                          dewpoint=dewpoint)
            out.append(xarray.Dataset(data_vars={'cape': surface_cape.m, 'cin': surface_cin.m},
                                      coords={'latitude': lat, 'longitude': lon}))
            
            out[-1] = out[-1].expand_dims({'longitude': [out[-1].longitude]})
            out[-1] = out[-1].expand_dims({'latitude': [out[-1].latitude]})
    
    out = xarray.merge(out)   
    return(out)
    
def surface_cape_vector(dat):
    """
    Use xarray implementations to calculate CAPE and CIN for each point.
    
    Arguments:
        dat: An xarray Dataset containing pressure, temperature, and 
             specific humidity.
            
    Returns: Dataset containing CAPE and CIN values.
    """
    
    # Dewpoints.
    dewpoint = metpy.calc.dewpoint_from_specific_humidity(pressure=dat.pressure,
                                                          temperature=dat.temperature,
                                                          specific_humidity=dat.specific_humidity)
    dewpoint = dewpoint.metpy.convert_units('K')
    dewpoint = dewpoint.metpy.dequantify()
    
    # CAPE and CIN.
    out = parcel.surface_based_cape_cin(pressure=dat.pressure,
                                        temperature=dat.temperature, 
                                        dewpoint=dewpoint)
    
    return(out)

def conv_properties_metpy_serial(dat):
    """
    Calculate convective properties for a set of points, using metpy in serial.
    
    Arguments:
       dat: An xarray Dataset containing dewpoint, pressure, temperature, and 
            specific humidity.
            
    Returns: Dataset containing all tested convective properties.
    """
    
    out = []
    
    for lat in dat.latitude.values:
        for lon in dat.longitude.values:
            # Assign units to values.
            pres = dat.sel(latitude=lat, longitude=lon).pressure.values * units(dat.pressure.attrs['units'])
            temp = dat.sel(latitude=lat, longitude=lon).temperature.values * units(dat.temperature.attrs['units'])
            spec_hum = (dat.sel(latitude=lat, longitude=lon).specific_humidity.values *
                        units(dat.specific_humidity.attrs['units']))

            # Dew points in K.
            dewpoint = metpy.calc.dewpoint_from_specific_humidity(pressure=pres,
                                                                  temperature=temp,
                                                                  specific_humidity=spec_hum).to('K')

            # Properties of fully-mixed lowest 100 hPa parcel (mp = mixed parcel).
            mp_pres, mp_temp, mp_dewpoint = metpy.calc.mixed_parcel(pressure=pres, 
                                                                    temperature=temp, 
                                                                    dewpoint=dewpoint, 
                                                                    depth=100*units.hPa)
            
            # Remove values below top of mixed layer and add in the mixed layer values
            pres_prof = concatenate([mp_pres, pres[pres < (pres[0] - 100*units.hPa)]])
            temp_prof = concatenate([mp_temp, temp[pres < (pres[0] - 100*units.hPa)]])
            dew_prof = concatenate([mp_dewpoint, dewpoint[pres < (pres[0] - 100*units.hPa)]])
            p, t, td, mixed_profile = metpy.calc.parcel_profile_with_lcl(pressure=pres_prof, 
                                                                         temperature=temp_prof, 
                                                                         dewpoint=dew_prof)
            
            # Dry lapse rates.
            dry = metpy.calc.dry_lapse(pressure=pres, temperature=temp[0])
            
            # Moist lapse rate.
            moist = metpy.calc.moist_lapse(pressure=pres, 
                                           temperature=temp[0],
                                           reference_pressure=900*units.hPa)
            
            # Profile for surface parcel.
            surf_pres, surf_temp, surf_dewpoint, surface_profile = metpy.calc.parcel_profile_with_lcl(
                pressure=pres, temperature=temp, dewpoint=dewpoint)
                        
            # LCL for surface-based parcel.
            surface_lcl_pressure, surface_lcl_temp = metpy.calc.lcl(pressure=pres[0], 
                                                                    temperature=temp[0], 
                                                                    dewpoint=dewpoint[0])
            
            # LFC for surface-based parcel.
            surface_lfc_pressure, surface_lfc_temp = metpy.calc.lfc(pressure=surf_pres, 
                                                                    temperature=surf_temp, 
                                                                    dewpoint=surf_dewpoint,
                                                                    which='bottom',
                                                                    parcel_temperature_profile=surface_profile)
            
            # EL for surface-based parcel.
            surface_el_pressure, surface_el_temp = metpy.calc.el(pressure=surf_pres, 
                                                                 temperature=surf_temp,
                                                                 dewpoint=surf_dewpoint,
                                                                 parcel_temperature_profile=surface_profile,
                                                                 which='top')
            
            # Surface-based CAPE and CIN.
            surface_cape, surface_cin = metpy.calc.surface_based_cape_cin(pressure=pres,
                                                                          temperature=temp,
                                                                          dewpoint=dewpoint)
            
            # CAPE and CIN for parcel made of fully-mixed lowest 100 hPa. 
            mixed_cape, mixed_cin = metpy.calc.mixed_layer_cape_cin(pressure=pres, 
                                                                    temperature=temp, 
                                                                    dewpoint=dewpoint, 
                                                                    depth=100*units.hPa)
            
            # CAPE and CIN for the most unstable parcel profile in the lowest 300 hPa.
            max_cape, max_cin = metpy.calc.most_unstable_cape_cin(pressure=pres, temperature=temp, 
                                                                  dewpoint=dewpoint, depth=300*units.hPa)
            
            # Interpolate profiles to get temperatures at 500 hPa and 850 hPa.
            int_pres = [500, 850] * units.hPa
            int_mixed_profile = metpy.interpolate.log_interpolate_1d(int_pres, p, mixed_profile)
            int_temp = metpy.interpolate.log_interpolate_1d(int_pres, p, t)
            int_dewpoint = metpy.interpolate.log_interpolate_1d(int_pres, p, td)

            # Lifted index using fully-mixed lowest 100 hPa mixed parcel.
            lifted_index = metpy.calc.lifted_index(pressure=int_pres, 
                                                   temperature=int_temp, 
                                                   parcel_profile=int_mixed_profile)

            # Deep convection index for lowest 100 hPa mixed parcel.
            dci = (int_temp[1].to(units.degC) + 
                   int_dewpoint[1].to(units.degC).magnitude * units.delta_degC - 
                   lifted_index.to(units.delta_degC))
            
            out.append(xarray.Dataset(data_vars={'dewpoint': (['model_level_number'], dewpoint.m),
                                                 'mp_pressure': mp_pres.m,
                                                 'mp_temperature': mp_temp.m,
                                                 'mp_dewpoint': mp_dewpoint.m,
                                                 'dry_lapse_temp': (['model_level_number'], dry.m),
                                                 'moist_lapse_temp': (['model_level_number'], moist.m),
                                                 'surface_profile': (['model_level_number_lcl'], surface_profile.m),
                                                 'surface_lfc_pressure': surface_lfc_pressure.m,
                                                 'surface_lcl_pressure': surface_lcl_pressure.m,
                                                 'surface_lcl_temp': surface_lcl_temp.m,
                                                 'surface_lfc_temp': surface_lfc_temp.m,
                                                 'surface_el_pressure': surface_el_pressure.m,
                                                 'surface_el_temp': surface_el_temp.m,
                                                 'surface_cape': surface_cape.m,
                                                 'surface_cin': surface_cin.m,
                                                 'surf_pres': (['model_level_number_lcl'], surf_pres.m),
                                                 'surf_temp': (['model_level_number_lcl'], surf_temp.m),
                                                 'mixed_cape': mixed_cape.m,
                                                 'mixed_cin': mixed_cin.m,
                                                 'max_cape': max_cape.m,
                                                 'max_cin': max_cin.m,
                                                 'lifted_index': float(lifted_index.m),
                                                 'dci': float(dci.m)},
                                      coords={'model_level_number': dat.pressure.model_level_number,
                                              'latitude': lat, 
                                              'longitude': lon}))
            
            out[-1] = out[-1].expand_dims({'longitude': [out[-1].longitude]})
            out[-1] = out[-1].expand_dims({'latitude': [out[-1].latitude]})
    
    out = xarray.merge(out)   
    return(out)

def conv_properties(dat, vert_dim='model_level_number', virt_temp=False):
    """
    Calculate convective properties for a set of points, using vectorised code.
    
    Arguments:
       dat: An xarray Dataset containing dewpoint, pressure, temperature, and 
            specific humidity.
       vert_dim: Name of the vertical dimension in dat.
       virt_temp: Use virtual temperature correction?
            
    Returns: Dataset containing all tested convective properties.
    """
      
    # Calculate dewpoints.
    dat['dewpoint'] = metpy.calc.dewpoint_from_specific_humidity(pressure=dat.pressure,
                                                                 temperature=dat.temperature,
                                                                 specific_humidity=dat.specific_humidity)
    dat['dewpoint'] = dat.dewpoint.metpy.convert_units('K')
    dat['dewpoint'] = dat.dewpoint.metpy.dequantify()
    
    # Mix the lowest 100 hPa.
    mp = parcel.mixed_parcel(pressure=dat.pressure, temperature=dat.temperature, dewpoint=dat.dewpoint)
    mp = mp.rename({'pressure': 'mp_pressure',
                    'temperature': 'mp_temperature',
                    'dewpoint': 'mp_dewpoint'})

    # Surface-based dry adiabats at each point.
    dry = parcel.dry_lapse(pressure=dat.pressure,
                           parcel_temperature=dat.temperature.isel(model_level_number=0))
    dry.name = 'dry_lapse_temp'
    
    # Surface-based moist adiabats at each point.
    moist = parcel.moist_lapse(pressure=dat.pressure,
                               parcel_temperature=dat.temperature.isel(model_level_number=0),
                               parcel_pressure=900)
    moist.name = 'moist_lapse_temp'
        
    # Mixed-parcel CAPE and CIN.
    mixed_cape_cin, mixed_profile = parcel.mixed_layer_cape_cin(pressure=dat.pressure,
                                                                temperature=dat.temperature, 
                                                                dewpoint=dat.dewpoint,
                                                                depth=100, return_profile=True,
                                                                virtual_temperature_correction=virt_temp)
    mixed_cape_cin = mixed_cape_cin.rename({'cape': 'mixed_cape',
                                            'cin': 'mixed_cin'})
    
    # CAPE and CIN for most unstable parcel in lowest 300 hPa.
    max_cape_cin = parcel.most_unstable_cape_cin(pressure=dat.pressure,
                                                 temperature=dat.temperature, 
                                                 dewpoint=dat.dewpoint,
                                                 depth=300,
                                                 virtual_temperature_correction=virt_temp)
    max_cape_cin = max_cape_cin.rename({'cape': 'max_cape',
                                        'cin': 'max_cin'})
    
    # Profile including LCL for surface-based parcel ascent.
    surface_profile = parcel.parcel_profile_with_lcl(pressure=dat.pressure,
                                                     temperature=dat.temperature,
                                                     dewpoint=dat.dewpoint,
                                                     parcel_temperature=dat.temperature.isel({vert_dim: 0}),
                                                     parcel_pressure=dat.pressure.isel({vert_dim: 0}),
                                                     parcel_dewpoint=dat.dewpoint.isel({vert_dim: 0}))
    
    # LFC for surface-based parcel.
    surface_lfc_el = parcel.lfc_el(pressure=surface_profile.pressure,
                                   parcel_temperature=surface_profile.temperature, 
                                   temperature=surface_profile.environment_temperature, 
                                   lcl_pressure=surface_profile.lcl_pressure, 
                                   lcl_temperature=surface_profile.lcl_temperature)
    surface_lfc_el = surface_lfc_el.rename({'lfc_pressure': 'surface_lfc_pressure',
                                            'lfc_temperature': 'surface_lfc_temp',
                                            'el_pressure': 'surface_el_pressure',
                                            'el_temperature': 'surface_el_temp'})
    
    # Surface-based CAPE and CIN.
    surface_cape_cin = parcel.surface_based_cape_cin(pressure=dat.pressure,
                                                     temperature=dat.temperature, 
                                                     dewpoint=dat.dewpoint,
                                                     virtual_temperature_correction=virt_temp)
    surface_cape_cin = surface_cape_cin.rename({'cape': 'surface_cape',
                                                'cin': 'surface_cin'})
    
    # Lifted index using mixed layer profile.
    lifted_index = parcel.lifted_index(profile=mixed_profile)
    
    # Deep convective index for mixed layer profile.
    dci = parcel.deep_convective_index(pressure=dat.pressure, temperature=dat.temperature,
                                       dewpoint=dat.dewpoint, lifted_index=lifted_index.lifted_index)
    
    # Rename clashing variables.
    surface_profile = surface_profile.rename({'pressure': 'surf_pres',
                                              'temperature': 'surface_profile',
                                              'lcl_pressure': 'surface_lcl_pressure',
                                              'lcl_temperature': 'surface_lcl_temp',
                                              'environment_temperature': 'surf_temp'})
    surface_profile = surface_profile.rename({'model_level_number': 'model_level_number_lcl'})
    
    out = xarray.merge([dat.dewpoint,
                        mp,
                        dry, 
                        moist,
                        mixed_cape_cin,
                        max_cape_cin,
                        surface_profile,
                        surface_lfc_el,
                        surface_cape_cin,
                        lifted_index,
                        dci])
    
    return out

def test_parcel_functions(dat, virt_temp=False):
    """
    Test that parcel functions in this module give the same results that MetPy gives for each profile.
    
    Arguments:
       dat: An xarray Dataset containing pressure, temperature, and specific humidity.
       virt_temp: Use the virtual temperature correction (MetPy does not).
       
    Returns: true if all tests passed, false if not.
    """
    
    print('Calculating xarray results...\t\t', end='')
    xarray_results, time = time_function(func=conv_properties, dat=dat, virt_temp=virt_temp)
    print(f'{str(time)} s.')
    
    print('Calculating metpy serial results...\t', end='')
    metpy_results, time = time_function(func=conv_properties_metpy_serial, dat=dat)
    print(f'{str(time)} s.')
  
    print(f'{"Differences":65} {"Max abs. diff":20} {"Max rel. diff":20}')
    for variable in metpy_results.keys():
        compare(xarray_results[variable], metpy_results[variable], name=variable)
          
    return xarray_results, metpy_results

def benchmark_cape(dat, points=[2,4,8,16,32,64]):
    """
    Calculate CAPE and CIN using MetPy and xarray functions for a variety of numbers of 
    points and return processing times.
    
    Arguments:
       dat: An xarray Dataset containing pressure, temperature, and specific humidity.
       points: Number of points (offset from lat/long 0,0) to time.
       
    Returns: 
    """
    
    num_points = []
    xr_times = []
    sr_times = []
    
    for p in points:
        pts = dat.isel(latitude=slice(0, p), longitude=slice(0, p)).load()
        xr, xr_time = time_function(func=surface_cape_vector, dat=pts)
        sr, sr_time = time_function(func=surface_cape_serial, dat=pts)
    
        num_points.append(p*p)
        xr_times.append(xr_time)
        sr_times.append(sr_time)
    
    res = xarray.Dataset({'vector_time': ('pts', xr_times),
                          'serial_time': ('pts', sr_times)},
                         {'pts': num_points})
    return res
    
import numpy as np
import astropy.units as u

__all__ = ["find_nearest_bin", "create_angular_area_scaling"]

def find_nearest_bin(array, value):
    """
    Find nearest value in an array

    :param array: ndarray
        Array to search
    :param value: float
        Search value
    :return: float
        Nearest bin value
    """

    idx = (np.abs(array - value)).argmin()
    return array[idx]

def create_angular_area_scaling(offset_bins, max_viewcone_radius):
    # Argh this code is horrible, but need to account for the angular area contained in each offset bin
    offset_area_scale = {}

    if len(offset_bins) == 1:
        offset_area_scale[offset_bins[0].value] = 1
    else:
        def angular_area(rmin, rmax):
            return np.pi * rmax**2 - np.pi*rmin**2
        total_area = angular_area(0*u.deg, max_viewcone_radius)             
        
        i=0
        imax=offset_bins.shape[0]-1
        diff = np.diff(offset_bins)/2

        for offset in offset_bins:

            upper_bound = offset + diff
            if i<imax:
                upper_bound = offset + diff
            lower_bound = 0
            if i>0:
                lower_bound = offset-diff
            
            print(upper_bound, lower_bound)
            ring_area = angular_area(lower_bound, upper_bound)

            offset_area_scale[offset.value] = total_area / ring_area
            i += 1
    return offset_area_scale
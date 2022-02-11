import numpy as np
import astropy.units as u
from eventio import EventIOFile
from eventio.simtel import MCShower
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time

__all__ = ["find_nearest_bin", "create_angular_area_scaling", "poisson_likelihood_gaussian",
           "tensor_poisson_likelihood", "create_xmax_scaling", "xmax_expectation"]

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

def xmax_expectation(energy):
    return 300 + 93 * np.log10(energy)

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

def create_xmax_scaling(xmax_bins, offset_bins, array_pointing, filename):
    output_dict = {}
    shower_count = 0

    with EventIOFile(filename) as f:

        dummy_time = Time('2010-01-01T00:00:00', format='isot', scale='utc')

        for o in f:
            if isinstance(o, MCShower):

                mc_shower = o.parse()
                
                energy = mc_shower["energy"]
                xmax_exp = xmax_expectation(energy)
                zenith = (np.pi/2) - mc_shower['altitude']

                xmax = mc_shower["xmax"] / np.cos(zenith)
                xmax_bin = find_nearest_bin(xmax_bins, xmax-xmax_exp)

                shower_direction = SkyCoord(alt=mc_shower['altitude']*u.rad, 
                                            az=mc_shower['azimuth']*u.rad, 
                                            frame=AltAz(obstime=dummy_time))
                offset = array_pointing.separation(shower_direction).to(u.deg).value
                offset_bin = find_nearest_bin(offset_bins.value, offset)
#                print(offset, offset_bin, xmax, xmax_exp, xmax-xmax_exp, xmax_bin, np.rad2deg(zenith))
                key = xmax_bin, offset_bin
                if key in output_dict.keys():
                    output_dict[key] += 1
                else:
                    output_dict[key] = 1
                shower_count += 1

    for key in output_dict.keys():
        output_dict[key] = float(shower_count)/output_dict[key]

    return output_dict 

def poisson_likelihood_gaussian(image, prediction, spe_width=0.5, ped=1):

    image = np.asarray(image)
    prediction = np.asarray(prediction)
    spe_width = np.asarray(spe_width)
    ped = np.asarray(ped)
    
    sq = 1. / np.sqrt(2 * np.pi * (np.power(ped, 2)
                                + prediction * (1 + np.power(spe_width, 2))))
    
    diff = np.power(image - prediction, 2.)
    denom = 2 * (np.power(ped, 2) + prediction * (1 + np.power(spe_width, 2)))
    expo = np.asarray(np.exp(-1 * diff / denom))
    
    # If we are outside of the range of datatype, fix to lower bound
    min_prob = np.finfo(expo.dtype).tiny
    expo[expo < min_prob] = min_prob
    
    return -2 * np.log(sq * expo)

def tensor_poisson_likelihood(image, prediction, spe_width=0.5, ped=1):
    import keras.backend as K
    import tensorflow as tf

    prediction = tf.clip_by_value(prediction, 1e-6, 1e9)

    sq = 1. / K.sqrt(2. * np.pi * (K.square(ped)
                                + prediction * (1. + K.square(spe_width))))

    diff = K.square(image - prediction)
    denom = 2. * (K.square(ped) + prediction * (1 + K.square(spe_width)))
    expo = K.exp(-1 * diff / denom)
    expo = tf.clip_by_value(expo, 1e-20, 100)

    return K.mean(-2 * K.log(sq * expo))

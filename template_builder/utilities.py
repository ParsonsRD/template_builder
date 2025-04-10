import numpy as np
import astropy.units as u
from eventio import EventIOFile
from eventio.simtel import MCShower
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time

__all__ = [
    "find_nearest_bin",
    "create_angular_area_scaling",
    "poisson_likelihood_gaussian",
    "tensor_poisson_likelihood",
    "create_xmax_scaling",
    "xmax_expectation",
    "rotate_translate",
]


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


def rotate_translate(pixel_pos_x, pixel_pos_y, x_trans, y_trans, phi):
    """
    Function to perform rotation and translation of pixel lists.

    Parameters
    ----------
    pixel_pos_x: np.array
        Array of pixel x positions
    pixel_pos_y: np.array
        Array of pixel x positions
    x_trans: float
        Translation of position in x coordinates
    y_trans: float
        Translation of position in y coordinates
    phi: float
        Rotation angle of pixels
    Returns
    -------
    np.array, np.array
        Transformed pixel x and y coordinates
    """

    cosine_angle = np.cos(phi[..., np.newaxis])
    sin_angle = np.sin(phi[..., np.newaxis])

    pixel_pos_trans_x = (x_trans - pixel_pos_x) * cosine_angle - (
        y_trans - pixel_pos_y
    ) * sin_angle

    pixel_pos_trans_y = (pixel_pos_x - x_trans) * sin_angle + (
        pixel_pos_y - y_trans
    ) * cosine_angle

    return pixel_pos_trans_x, pixel_pos_trans_y


def xmax_expectation(energy):
    """
    Expected slant depth of shower maximum for gamma rays
    as a function of energy.

    Parameters
    ----------
    energy : float
        Gamma ray energy in TeV

    Returns
    -------
    float
        Expected slant depth of shower maximum.
    """
    return 300 + 93 * np.log10(energy)


def create_angular_area_scaling(offset_bins, max_viewcone_radius):
    """
    Scale offset bins for angular area

    Parameters
    ----------
    offset_bins : _type_
        _description_
    max_viewcone_radius : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    offset_area_scale = {}

    if len(offset_bins) == 1:
        offset_area_scale[offset_bins[0].value] = 1
    else:

        def angular_area(rmin, rmax):
            return np.pi * rmax**2 - np.pi * rmin**2

        total_area = angular_area(0 * u.deg, max_viewcone_radius)

        i = 0
        imax = offset_bins.shape[0] - 1
        diff = np.diff(offset_bins) / 2

        for offset in offset_bins:

            upper_bound = offset + diff
            if i < imax:
                upper_bound = offset + diff
            lower_bound = 0
            if i > 0:
                lower_bound = offset - diff

            print(upper_bound, lower_bound)
            ring_area = angular_area(lower_bound, upper_bound)

            offset_area_scale[offset.value] = total_area / ring_area
            i += 1
    return offset_area_scale


def create_xmax_scaling(xmax_bins, offset_bins, array_pointing, filename):
    """
    Count fraction of simulated evemts that fall in the different xmax, offset bins.
    Relevant for the computation of the trigger fraction templates.

    Parameters
    ----------
    xmax_bins : np.array
        Bin edges of the xmax bins (relative to the expectation value)
    offset_bins : np.array
        Bin edges of the bins in FoV offset 
    array_pointing : SkyCoord
        Pointing direction of the simulated telescope array
    filename : Path
        Path to the input simulation file

    Returns
    -------
    Dictionary
        Fraction of events in a given xmax, offset bin combination
    """
    output_dict = {}
    shower_count = 0

    with EventIOFile(filename) as f:

        dummy_time = Time("2010-01-01T00:00:00", format="isot", scale="utc")

        for o in f:
            if isinstance(o, MCShower):

                mc_shower = o.parse()

                energy = mc_shower["energy"]
                xmax_exp = xmax_expectation(energy)
                zenith = (np.pi / 2) - mc_shower["altitude"]

                xmax = mc_shower["xmax"] / np.cos(zenith)
                xmax_bin = find_nearest_bin(xmax_bins, xmax - xmax_exp)

                shower_direction = SkyCoord(
                    alt=mc_shower["altitude"] * u.rad,
                    az=mc_shower["azimuth"] * u.rad,
                    frame=AltAz(obstime=dummy_time),
                )
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
        output_dict[key] = float(shower_count) / output_dict[key]

    return output_dict


def poisson_likelihood_gaussian(image, prediction, spe_width=0.5, ped=1):
    """
    Twice the negative logarithm of the Gaussian ImPACT likelihood function.
    Used as a loss function in the MLP fit.

    Parameters
    ----------
    image : list
        Measured pixel charges
    prediction : list
        Expected pixel charges
    spe_width : float, optional
        Parameter controlling the width of the likelihood at high amplitudes, by default 0.5
    ped : float, optional
        Pedestal width parameter controlling the width of the likelihood at low amplitudes, by default 1

    Returns
    -------
    np.array
        Twice the negative log likelihood
    """

    image = np.asarray(image)
    prediction = np.asarray(prediction)
    spe_width = np.asarray(spe_width)
    ped = np.asarray(ped)

    sq = 1.0 / np.sqrt(
        2 * np.pi * (np.power(ped, 2) + prediction * (1 + np.power(spe_width, 2)))
    )

    diff = np.power(image - prediction, 2.0)
    denom = 2 * (np.power(ped, 2) + prediction * (1 + np.power(spe_width, 2)))
    expo = np.asarray(np.exp(-1 * diff / denom))

    # If we are outside of the range of datatype, fix to lower bound
    min_prob = np.finfo(expo.dtype).tiny
    expo[expo < min_prob] = min_prob

    return -2 * np.log(sq * expo)


def tensor_poisson_likelihood(image, prediction, spe_width=0.5, ped=1):
    """
    A tensorflow implementation of the negative log ImPACT likelihood

        Parameters
    ----------
    image : list
        Measured pixel charges
    prediction : list
        Expected pixel charges
    spe_width : float, optional
        Parameter controlling the width of the likelihood at high amplitudes, by default 0.5
    ped : float, optional
        Pedestal width parameter controlling the width of the likelihood at low amplitudes, by default 1

    Returns
    -------
    keras tensor
        Twice the negative log likelihood
    """

    import keras.backend as K
    import tensorflow as tf

    prediction = tf.clip_by_value(prediction, 1e-6, 1e9)

    sq = 1.0 / K.sqrt(
        2.0 * np.pi * (K.square(ped) + prediction * (1.0 + K.square(spe_width)))
    )

    diff = K.square(image - prediction)
    denom = 2.0 * (K.square(ped) + prediction * (1 + K.square(spe_width)))
    expo = K.exp(-1 * diff / denom)
    expo = tf.clip_by_value(expo, 1e-20, 100)

    return K.mean(-2 * K.log(sq * expo))

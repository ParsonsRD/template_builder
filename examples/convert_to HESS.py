import gzip
import pickle
import numpy as np
from rootpy.plotting import Hist2D
from rootpy.io import root_open
from operator import itemgetter, attrgetter
import math
from root_numpy import array2hist
import sys

from tqdm import tqdm


def round_sigfigs(num, sig_figs):
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0

def rollover_round(azimuth):
    azimuth = round(azimuth)
    if azimuth == 360:
        return 0
    return azimuth


def convert_file(ctapipe_file, hap_file, fl_scale, pixel_area, bounds, scale_factor):
    # This value is from CT5
    fl_scale = 1./fl_scale

    myfile = root_open(hap_file, 'recreate')
    templates = pickle.load(gzip.open(ctapipe_file,"r"))

    keys = list(templates.keys())
    keys = sorted(keys, key=lambda element: (rollover_round(element[1]), round(element[0]),  element[2],  element[3],  element[4]))

    os = 0.01
    for key in tqdm(keys):
        zen, az, en, impact, xmax = key

        t = templates[key]
        shape = t.shape
        half_bin_width_0 = (float(bounds[0][0]) - float(bounds[0][1]))/float(shape[0] * 2)
        half_bin_width_1 = (float(bounds[1][0]) - float(bounds[1][1]))/float(shape[1] * 2)

        name_string = "template_"
        name_string += str(rollover_round(int(az))) + "azm_"
        name_string += str(int(round(zen))) + "deg_0.0off_"
        name_string += str(round_sigfigs(en, 5)) + "TeV_"
        name_string += str(int( impact)) + "m_"
        name_string += str(int(xmax/25.) + 100) + "XSmooth"

        t[np.isnan(t)] = 0
        t[t<0] = 0

        hist = Hist2D(shape[0],(bounds[0][0]-half_bin_width_0)/fl_scale,(bounds[0][1]+half_bin_width_0)/fl_scale,
                      shape[1], (bounds[1][0]-half_bin_width_1)/fl_scale,(bounds[1][1]+half_bin_width_1)/fl_scale, 
                      name=name_string)#was mult

        array2hist((t)/pixel_area, hist)

        hist.SetDirectory(myfile)

    myfile.Write()

def main():

    ctapipe_file = sys.argv[1]
    hap_file = sys.argv[2]
    fl_scale = sys.argv[3]
    pixel_area =  sys.argv[4]
    scale_factor =  sys.argv[5]

    convert_file(ctapipe_file, hap_file, fl_scale, pixel_area, scale_factor)
    
if __name__ == "__main__":
    main()

import numpy as np
from scipy.interpolate import interp1d

__all__ = ["extend_xmax_range", "extend_distance_range", "extend_template_coverage"]

def extend_xmax_range(xmax_bins, templates):
    """
    Copy templates in empty xmax bins, helps to prevent problems from reaching the
    edge of the interpolation space.

    :param templates: dict
        Image templates
    :return: dict
        Extended image templates
    """

    # Create dictionary for our new templates
    extended_templates = dict()

    # Loop over image templates
    for key in templates:
        min_key_bin = list()
        key_copy = 0
        # For each entry loop forward over possible xmax entries to check if they
        # exist
        for xb in xmax_bins:
            key_test = (key[0], key[1], key[2], key[3], xb, key[5])
            # keep looping until we have found the largest xmax value
            if (key_test not in extended_templates.keys()) and \
                    (key_test not in templates.keys()):
                min_key_bin.append(key_test)
            else:
                key_copy = key_test
                break
        # Then copy in the highest xmax valid template into these etries
        for k in min_key_bin:
            if key_copy != 0:
                extended_templates[k] = templates[key_copy]

        min_key_bin = list()
        key_copy = 0
        # Now we just do the same in reverse
        for xb in reversed(xmax_bins):
            key_test = (key[0], key[1], key[2], key[3], xb, key[5])
            if (key_test not in extended_templates.keys()) and \
                    (key_test not in templates.keys()):
                min_key_bin.append(key_test)
            else:
                key_copy = key_test
                break

        for k in min_key_bin:
            if key_copy != 0:
                extended_templates[k] = templates[key_copy]

    # Copy new template entries into the original
    templates.update(extended_templates)

    return templates

def extend_distance_range(xmax_bins, templates, additional_bins=4):
    """
    Copy templates in empty xmax bins, helps to prevent problems from reaching the
    edge of the interpolation space.

    :param templates: dict
        Image templates
    :return: dict
        Extended image templates
    """
    keys = np.array(list(templates.keys()))
    if len(list(templates.keys())) < 1:
        return templates

    distances = np.sort(np.unique(keys.T[3]))
    energies = np.unique(keys.T[2])
    zeniths = np.unique(keys.T[0])
    azimuths = np.unique(keys.T[1])
    offsets = np.unique(keys.T[5])
    
    extended_templates = dict()
    for zen in zeniths:
        for az in azimuths:
            for en in energies:
                for xmax in xmax_bins:
                    for off in offsets:

                        i = 0
                        distance_list = list()

                        # If we have no template at 0 copy the lowest value
                        if distances[0] != 0.:
                            copied = False
                            for d in distances:
                                key = (zen, az, en, d, xmax, off)
                                if key in templates.keys() and not copied:
                                    extended_templates[(zen, az, en, 0, xmax, off)] = \
                                        templates[key]
                                    copied = True

                        for dist in distances[0:]:
                            key = (zen, az, en, dist, xmax, off)
                            if key not in templates.keys():
                                break
                            else:
                                distance_list.append(templates[key])
                                i += 1

                        num_dists = len(distance_list)
                        if num_dists > 1 and num_dists < len(distances):
                            distance_list = np.array(distance_list)

                            diff = len(distances) - len(distance_list)

                            if diff > additional_bins:
                                diff = additional_bins
                            for j in range(i, i + diff):
                                lower = i-3;
                                if lower < 0:
                                    lower = 0
                                interp = interp1d(distances[0:i], distance_list, axis=0,
                                                    bounds_error=False,
                                                    fill_value="extrapolate", kind="linear")

                                int_val = interp(distances[j])

                                int_val[int_val < 0] = 0
                                key = (zen, az, en, distances[j], xmax, off)

                                extended_templates[key] = int_val

    templates.update(extended_templates)

    return templates

def extend_template_coverage(xmax_bins, templates):

    templates = extend_xmax_range(xmax_bins, templates)
    templates = extend_distance_range(xmax_bins, templates, 4)

    return templates

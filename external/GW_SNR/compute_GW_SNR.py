"""
.. module:: compute_SNR
    :synopsis: Compute SNR for GWB monopole
.. moduleauthor:: Florian Schulze <florian.tobias.schulze@rwth-aachen.de>

This module provides functions to calculate the SNR for the detection of a
CGWB monopole Omega_GW, given a detctor network.

"""
import numpy as np
from scipy.interpolate import interp1d

def read_psds(det_filename, freqs):
    """Reads a PSD file and caluclates the PSD at the requested frequencies freqs

    Args:
        det_filename (str): PSD file
        freqs (array): requested frequencies for the PSD

    Returns:
        array: PSD of the dector at the frequencies freqs
    """
    interpolation_type = 'quadratic'

    data = np.genfromtxt(det_filename).T
    detector_frequencies = data[0]
    detector_psd = data[1]
    f = interp1d(detector_frequencies, detector_psd, kind=interpolation_type)

    return f(freqs)


def get_gw_detector_psd(detector, freqs):
    """Calulates the PSD for a detector network

    Args:
        detector (str or list): Detector name or list with the PSD files of the detector (e.g. "CE+ET")
        freqs (array): requested frequencies for the PSD

    Returns:
        nd_array: detecor_psd
    """
    detector_num = 1
    psd_files = []

    if (detector == "CE+ET"):
        detector_num = 5
        psd_files = ["./Detector_PSDs/ce1.txt", "./Detector_PSDs/ce2.txt", "./Detector_PSDs/ET.txt", "./Detector_PSDs/ET.txt", "./Detector_PSDs/ET.txt"]
    
    else:
        psd_files = detector
        detector_num = len(psd_files)

    detector_psd = np.empty((detector_num, len(freqs)))
    
    for in_det in range(detector_num):
        detector_psd[in_det] = read_psds(psd_files[in_det], freqs)
    
    return detector_psd


def compute_GW_SNR(freqs, Omega_GW, detector_psd, h=0.67, T_obs=10):
    """Calculates the SNR for the CGWB monopole Omega_GW given an detector detector_psd.

    Args:
        freqs (array): frequencies in Hz
        Omega_GW (array): CGWB monopole
        detector_psd (nd_array): detector PSD
        h (int, optional): reduced Hubble rate h. Defaults to 0.67.
        T_obs (int, optional): observation time in years. Defaults to 10.

    Returns:
        float: Signal to Noise Ratio (SNR)
    """
    snr = 0
    
    for in_det1 in range(len(detector_psd)):
        for in_det2 in range(len(detector_psd)):
            f1 = np.power(Omega_GW[:-1], 2.) \
                    / np.power(detector_psd[in_det1][:-1] * detector_psd[in_det2][:-1], 2.) \
                    / np.power(freqs[:-1], 6.)
            f2 = np.power(Omega_GW[1:], 2.) \
                    / np.power(detector_psd[in_det1][1:] * detector_psd[in_det2][1:], 2.) \
                    / np.power(freqs[1:], 6.)
            dx = (freqs[1:]-freqs[:-1])/2.
            
            x = np.power( np.power(h*3.24*1e-18, 2.) / (4.*np.pi**2 * np.sqrt(4.*np.pi)), 2.) * (f1+f2) * dx
            snr += np.sum(x)

    snr = np.sqrt(3.15e7*T_obs*snr)
    
    return snr


def main():
    from classy import Class
    import matplotlib.pyplot as plt

    omega_freq_size=1000
    f_min=5.01
    f_max=1000

    M = Class()
    M.set({
        'output': 'OmGW, gwCl',
        'gwb_source_type':  'PBH_gwb',
        'f_pivot':          25,
        'f_min':            f_min,
        'f_max':            f_max,
        'A_star':           2e-5,
        'f_star':           100.,
        'f_NL':             1.,
        'tau_ini_gwb':      0.1,
    })

    M.compute()

    OmGW = M.get_omega_gw()
    freqs = OmGW['f [Hz]']
    Omega_GW = OmGW['Omega_GW(f)']

    freqs = np.linspace(f_min, f_max, omega_freq_size)
    freqs = np.geomspace(f_min, f_max, omega_freq_size)
    Omega_GW = np.array([M.Omega_GW(f) for f in freqs])
    omega_freq_size = len(freqs)
    h = M.h()

    # plt.figure()
    # plt.loglog(freqs, Omega_GW)

    detecor_psd = get_gw_detector_psd("CE+ET", freqs)

    snr = compute_GW_SNR(freqs, Omega_GW, detecor_psd, h)
    print('SNR = %g' % snr)

    plt.show()

    return 0


if __name__ == '__main__':
    main()

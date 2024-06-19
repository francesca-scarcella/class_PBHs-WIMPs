from __future__ import absolute_import, division, print_function

import numpy as np
import os
import sys
import dill
from scipy.special import erf
from scipy.integrate import quad, dblquad, trapz
from scipy.interpolate import interp1d




from .__init__ import DarkAgesError as err
data_dir = os.path.join( os.path.dirname(os.path.realpath( __file__ )), 'data' )

def boost_factor_halos(redshift,zh,fh):
    ret = 1 + fh*erf(redshift/(1+zh))/redshift**3
    return ret
# def boost_factor_clump(redshift,NEWPARM):
#     ret = 1 + fh*erf(redshift/(1+zh))/redshift**3
#     return ret

#///////////////////////DEFINE NEW BOOST FACTOR FOR ANNIHILATION IN PBH SPIKES///////////////////////////

def boost_factor_spikes_vec(z_vec,MPBH,fPBH, mX, sigmav, spikeprofiles):


    return np.asarray([boost_factor_spikes_from_file(z_iterator, MPBH, fPBH, mX, sigmav, spikeprofiles) for z_iterator in z_vec])



def boost_factor_spikes_from_file(z,MPBH,fPBH, mX, sigmav, spikeprofiles):

    """

     MPBH : in Msun
     mX: DM mass in GeV
     sigmav: in cm3/s
     rho_cdm_today_kg: in kg/m3
     spikeprofiles: 1 for us, 2 for Carr

     Returns the boost factor at z (dimensionless)

    """



    c= 299792e3        # m/s
    Msun= 1.98847e30   # kg

    #convert the PBH mass in grams to compute the boost factor
    MPBH_g=MPBH*Msun*1e3

    # I need to define the cosmology to obtain t(z)
    # We will have to check the error made by keeping the cosmological parameters fixed in this module

    h = 0.678
    H0 = 100.0*h #(km/s) Mpc^-1
    Omegac = 0.259;
    Omega_DM = 0.1186/(h**2)
    Omega_L = 0.692
    Omega_m = 0.308
    Omega_r = 9.3e-5
    rhoc = 9e-27 ; #kg/m^3
    rho_cdm_today_kg= rhoc*Omegac
    rho_cdm_today_g= rhoc*Omegac/1e3 #converting to g/cm^3
    H0_peryr = 67.8*(3.24e-20)*(60*60*24*365)
    ageUniverse = 13.799e9 #y

    Hubble= H0_peryr*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)
    t= quad(lambda x: 1.0/((1+x)*H0_peryr*np.sqrt(Omega_L + Omega_m*(1+x)**3 + Omega_r*(1+x)**4)), z, np.inf)[0] *3.154e7    #time in seconds
    #note: the integration failes above z=10^5, ok since the table of z values is up to 10^4 only. Add an error message if z>10^4 is used


    #convert the DM mass in grams
    mX_g=mX*1.602e-10/c**2*1e3          #  1GeV=1.602*1e-10 J

    # obtain the maximum density in g/cm3:
    rhomax_of_z= mX_g/(sigmav*t)


    ######### READ THE J-FACTORS FROM FILE ##############

    #generate the name of the file in the form:

       #Jfactors_mX_GeV_Xkd_MPBH.dat
       #example: Jfactors_100_GeV_10135_1e2.dat


    #dir_path="./Jfactor_files/"
    dir_path=os.path.join(data_dir, 'Jfactor_files')

    #print("dir_path=", dir_path)

    if spikeprofiles==1:
        basefilename="/Jfactors_"
    elif spikeprofiles==2:
        basefilename="/Jfactors_Carr_"

    mXstring="{:.0f}_GeV_".format(mX)

    xkdstring="10135_"    #I have only this option for now

    MPBHexpstr = "{:.0f}".format(abs(np.log10(MPBH)))

    if (np.log10(MPBH)==0):
        MPBHstring="1"
    elif (np.log10(MPBH)>0):
        MPBHstring="1e"+MPBHexpstr
    elif (np.log10(MPBH)<0):
        MPBHstring="1em"+MPBHexpstr


    if spikeprofiles==1:
        filename=basefilename+mXstring+xkdstring+MPBHstring+".dat"
    elif spikeprofiles==2:
        filename=basefilename+mXstring+MPBHstring+".dat"
    

    #print("reading J factors from the file: ",dir_path + filename)

    """
    Skip the lines tha are blank or commented;
    the first line is the number of lines (needed for the C code)
    the first line after that is the density of the spike at r_s and the J factor for the whole spike
    the last line is the minimum density of the spike and corresponding J factor (close to 0)

    """

    rhomax = []
    Jfactors= []


    with open(dir_path + filename, "r") as file:
        for line in file:
            # Skip commented or blank lines
            if line.startswith("#") or line.strip() == '':
                continue
            # Check if nlines is not yet defined
            if 'nlines' not in locals():
                # Store the first non-commented, non-blank line in nlines
                nlines = line.strip()
            else:
                # Split the line into two parts and store them in separate arrays
                parts = line.strip().split()
                rhomax.append(parts[0])
                Jfactors.append(parts[1])

    rhomax = np.array(rhomax, dtype=float)
    Jfactors = np.array(Jfactors, dtype=float)

    #print("nlines is", nlines)
    #nlines is not necessary here
    del(nlines)


    ########## INTERPOLATE THE J-FACTORS ###########

    """
     Setting the extremes:
    - if rhomax_of_z is larger than the first entry in the first column, the spike has not been depleted yet: we use the J-factor for the whole spike
    - if rhomax_of_z is smaller than the first entry in the first column we set the J-factor to zero
    """

    J_interp = interp1d(rhomax, Jfactors, kind='linear', bounds_error=False, fill_value=(0, Jfactors[0]))


    ######## COMPUTE J(z) and the boost factor ###########

    J_of_z = J_interp(rhomax_of_z)

    boost_factor = J_of_z * fPBH /rho_cdm_today_g/(1+z)**3/(MPBH_g)



    return boost_factor



















def secondaries_from_cirelli(logEnergies,mass,primary, **DarkOptions):
    from .common import sample_spectrum
    cirelli_dir = os.path.join(data_dir, 'cirelli')
    dumpername = 'cirelli_spectrum_of_{:s}.obj'.format(primary)

    injection_history = DarkOptions.get("injection_history","annihilation")
    if "decay" in injection_history:
        equivalent_mass = mass/2.
    else:
        equivalent_mass = mass
    if equivalent_mass < 5 or equivalent_mass > 1e5:
        raise err('The spectra of Cirelli are only given in the range [5 GeV, 1e2 TeV] assuming DM annihilation. The equivalent mass for the given injection_history ({:.2g} GeV) is not in that range.'.format(equivalent_mass))

    if not hasattr(logEnergies,'__len__'):
        logEnergies = np.asarray([logEnergies])
    else:
        logEnergies = np.asarray(logEnergies)

    if not os.path.isfile( os.path.join(cirelli_dir, dumpername)):
        sys.path.insert(1,cirelli_dir)
        from spectrum_from_cirelli import get_cirelli_spectra
        masses, log10X, dNdLog10X_el, dNdLog10X_ph, dNdLog10X_oth = get_cirelli_spectra(primary)
        total_dNdLog10X = np.asarray([dNdLog10X_el, dNdLog10X_ph, dNdLog10X_oth])
        from .interpolator import NDlogInterpolator
        interpolator = NDlogInterpolator(masses, np.rollaxis(total_dNdLog10X,1), exponent = 0, scale = 'log-log')
        dump_dict = {'dNdLog10X_interpolator':interpolator, 'log10X':log10X}
        with open(os.path.join(cirelli_dir, dumpername),'wb') as dump_file:
            dill.dump(dump_dict, dump_file)
    else:
        with open(os.path.join(cirelli_dir, dumpername),'rb') as dump_file:
            dump_dict = dill.load(dump_file)
            interpolator = dump_dict.get('dNdLog10X_interpolator')
            log10X = dump_dict.get('log10X')
    del dump_dict
    temp_log10E = log10X + np.log10(equivalent_mass)*np.ones_like(log10X)
    temp_el, temp_ph, temp_oth = interpolator.__call__(equivalent_mass) / (10**temp_log10E * np.log(10))[None,:]
    ret_spectra = np.empty(shape=(3,len(logEnergies)))
    ret_spectra = sample_spectrum(temp_el, temp_ph, temp_oth, temp_log10E, mass, logEnergies, **DarkOptions)
    return ret_spectra

def secondaries_from_simple_decay(E_secondary, E_primary, primary):
    if primary not in ['muon','pi0','piCh']:
        raise err('The "simple" decay spectrum you asked for (species: {:s}) is not (yet) known.'.format(primary))

    if not hasattr(E_secondary,'__len__'):
        E_secondary = np.asarray([E_secondary])
    else:
        E_secondary = np.asarray(E_secondary)

    decay_dir  = os.path.join(data_dir, 'simple_decay_spectra')
    dumpername = 'simple_decay_spectrum_of_{:s}.obj'.format(primary)
    original_data = '{:s}_normed.dat'.format(primary)

    if not os.path.isfile( os.path.join(decay_dir, dumpername)):
        data = np.genfromtxt( os.path.join(decay_dir, original_data), unpack = True, usecols=(0,1,2,3))
        from .interpolator import NDlogInterpolator
        spec_interpolator = NDlogInterpolator(data[0,:], data[1:,:].T, exponent = 1, scale = 'lin-log')
        dump_dict = {'spec_interpolator':spec_interpolator}
        with open(os.path.join(decay_dir, dumpername),'wb') as dump_file:
            dill.dump(dump_dict, dump_file)
    else:
        with open(os.path.join(decay_dir, dumpername),'rb') as dump_file:
            dump_dict = dill.load(dump_file)
            spec_interpolator = dump_dict.get('spec_interpolator')

    x = E_secondary / E_primary
    out = spec_interpolator.__call__(x)
    out /= (np.log(10)*E_secondary)[:,None]
    return out

def luminosity_accreting_bh(Energy,recipe,PBH_mass):
    if not hasattr(Energy,'__len__'):
        Energy = np.asarray([Energy])
    if recipe=='spherical_accretion':
        a = 0.5
        Ts = 0.4*511e3
        Emin = 1
        Emax = Ts
        out = np.zeros_like(Energy)
        Emin_mask = Energy > Emin
        # Emax_mask = Ts > Energy
        out[Emin_mask] = Energy[Emin_mask]**(-a)*np.exp(-Energy[Emin_mask]/Ts)
        out[~Emin_mask] = 0.
        # out[~Emax_mask] = 0.

    elif recipe=='disk_accretion':
        a = -2.5+np.log10(PBH_mass)/3.
        Emin = (10/PBH_mass)**0.5
        # print a, Emin
        Ts = 0.4*511e3
        out = np.zeros_like(Energy)
        Emin_mask = Energy > Emin
        out[Emin_mask] = Energy[Emin_mask]**(-a)*np.exp(-Energy[Emin_mask]/Ts)
        out[~Emin_mask] = 0.
        Emax_mask = Ts > Energy
        out[~Emax_mask] = 0.
    else:
        from .__init__ import DarkAgesError as err
        raise err('I cannot understand the recipe "{0}"'.format(recipe))
    # print out, Emax_mask
    return out/Energy #We will remultiply by Energy later in the code

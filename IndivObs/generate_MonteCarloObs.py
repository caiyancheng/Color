import numpy as np
import pandas as pd

def fnc_genMonteCarloObs(n_population, list_Age, fs):
    # Define variables
    list_paramNames = ['od_lens', 'od_macula', 'od_L', 'od_M', 'od_S', 'shft_L', 'shft_M', 'shft_S']
    stdDevAllParam = np.array([19.1, 37.2, 17.9, 17.9, 14.7, 4.0, 3.0, 2.5])
    stdDevAllParam[:2] *= 0.98
    stdDevAllParam[2:] *= 0.50

    # Normally-distributed physiological factors
    vAll = np.random.normal(scale=stdDevAllParam, size=(n_population, 8))

    # Generate random ages
    list_AgeRound = np.round(list_Age)
    var_age = np.random.choice(np.unique(list_AgeRound), n_population,
                               p=np.histogram(list_AgeRound, bins=np.unique(list_AgeRound))[0]/len(list_AgeRound))

    # File import
    files = {
        'rmd': pd.read_csv('cie2006_RelativeMacularDensity.txt'),
        'LMSa': pd.read_csv('cie2006_Alms.txt'),
        'docul': pd.read_csv('cie2006_docul.txt')
    }

    # LMS calculation
    LMS_All = np.empty((79, 3, n_population))
    LMS_All[:] = np.NaN

    for k in range(n_population):
        # You need to translate cie2006cmfsEx function to Python as well
        t_LMS, _, _, _ = cie2006cmfsEx(var_age[k], fs, *vAll[k, :], files)
        LMS_All[:, :, k] = t_LMS

    return LMS_All, var_age, vAll

# Note: You must translate 'fnc_MonteCarloParam' and 'cie2006cmfsEx' functions to Python as well.

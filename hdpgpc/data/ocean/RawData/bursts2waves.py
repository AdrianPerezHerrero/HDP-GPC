# -*- coding: utf-8 -*-
"""
A script to characterise wind wave conditions from pressure sensor data

Last update on 2022-11-30

@author: Zuzanna Swirad zswirad@igf.edu.pl
"""

# IMPORT PACKAGES
import os
import pandas as pd
import numpy as np
import math
from scipy.fft import fft

# SET PARAMETERS
lat = 77 # latitude (degrees)
Fs = 1 # sampling frequency (Hz)
T = 1/Fs # sampling period (s)
L = 1024 # number of samples in a single burst = time vector length
t = L*T # burst length (s)

# LOAD BURST DATA
# The raw data from IG PAS LONGHORN oceanographic monitoring
# (https://dataportal.igf.edu.pl/dataset/sea-pressure-and-wave-monitoring-datasets-in-hornsund-fjord) 
# were pre-processed in such a way that each file represents a singe deployment
# containing a series of pressure measurements at 1 Hz frequency (full seconds) 
# in 1024 s bursts (hh:00:00 to hh:17:03) starting at full hours UTC. The raw data 
# were trimmed to full days - they start at 00:00:00 and end at 23:17:03.
# There are 3 columns: 
    # 1. time ('yyyy-mm-dd hh:mm:ss'),
    # 2. burst number [1:n],
    # 3. raw pressure (dbar).

os.chdir(r'C:/waves/inputs') # change accordingly
FileName = 'HBK1.txt' # change accordingly
data = pd.read_csv(FileName, header = None) 
data.columns = ['Time', 'Burst', 'Pressure']
data['Time'] = pd.to_datetime(data['Time'])

# LOAD ATOMSPHERIC PRESSURE DATA
# Hourly atmospheric pressure at the sea level was downloaded from the Polish
# Polar Station monitoring website (https://monitoring-hornsund.igf.edu.pl/). 
# It is a single array of pressure (mbar) with the first row representing 
# 2013-07-21 at 00:00 UTC (time of the first pressure sensor deployment)
# followed by pressure at 1 hour interval until 2021-06-30 23:00 UTC.

StartTime = pd.to_datetime('2013-07-21 00:00:00')
AirPressureFile = 'C:/waves/inputs/airpressure.txt' # change accordingly
AirPressureSeries = pd.read_csv(AirPressureFile, header = None)

# PREPARE TABLES TO SAVE OUTPUTS
spectra = pd.DataFrame(index = range(data['Burst'].max()), columns = range(int(L/2)+1))
column_names = ['Burst', 'Time', 'MeanDepth', 'f_lastval', 'Hs', 'Tm_01', 'Tm_02', 
                'Tm_neg10', 'Hs_inf', 'Tm_01_inf', 'Tm_02_inf', 'Tm_neg10_inf']
properties = pd.DataFrame(index = range(data['Burst'].max()), columns = column_names)
spectraName = FileName[:-4] + '_spectra.txt'
propertiesName = FileName[:-4] + '_properties.txt'

# PROCESS DATA ON THE BURST-BY-BURST BASIS
for n in range(min(data['Burst']), max(data['Burst']) + 1):
    df = data[data['Burst'] == n]
    df = df.reset_index(drop = True)
    df_time = df.Time[0]
    TimeDifference = df.Time[0] - StartTime
    TimeDifference = int(TimeDifference.total_seconds()/3600) # (h)
    AirPressure = AirPressureSeries.loc[TimeDifference, 0]

    # CALCULATE SEA PRESSURE
    df['SeaPressure'] = pd.Series(np.zeros(df.shape[0] + 1))
    df['SeaPressure'] = df['Pressure'] - (AirPressure/100) # mbar2dbar

    # CALCULATE DEPTH FROM PRESSURE 
    # UNESCO formula (Fofonoff & Millard, 1983) under the assumption of 
    # constant water temperature 0 C and salinity 35 PSU
    x = math.sin(lat/57.29578)**2
    g = 9.780318*(
        1+(5.2788*10**-3+2.36*10**-5*x)
        *x)+1.092*10**-6*df['SeaPressure'] # acceleration due to gravity (m/s2)
    df['Depth'] = pd.Series(np.zeros(df.shape[0] + 1))
    df['Depth'] = ((((
        -1.82*10**-15*df['SeaPressure']+2.279*10**-10)
        *df['SeaPressure']-2.2512*10**-5)
        *df['SeaPressure']+9.72659)
        *df['SeaPressure'])/g # (m)

    # REMOVE THE SLOWLY-VARYING COMPONENT OF WATER DEPTH
    # (effect of e.g. sea level rise, tides and storm surges)
    # Subtract a fitted 2nd order polynomial trend from each burst
    time = range(L) # time vector [0:1:1023]
    tides = np.polyfit(time, df['Depth'], 2)
    S1 = df['Depth'] - np.polyval(tides, time) # depth variability associated with wind waves (m)
    
    # CALCULATE WAVE SPECTRA
    S1 = S1.to_numpy()
    Y = fft(S1) # Fast Fourier Transform
    f = np.r_[:int(L/2+1)]*Fs/L # frequency vector [0:d_f:0.5]
    d_f = f[1] - f[0] # frequency interval
    P2 = abs(Y)
    P1 = P2[:int(L/2+1)]**2/(Fs*L)
    P1[1:-1] = 2*P1[1:-1] # energy density spectrum at depth (m2 s)

    # APPLY DEPTH CORRECTION
    h = df['Depth'].mean() # mean depth (m)
    z = df['Depth'].mean() # logger depth (m)
    g = 9.780318*(
        1+(5.2788*10**-3+2.36*10**-5*x)
        *x)+1.092*10**-6 # acceleration due to gravity (m/s2)
    k = np.linspace(0,1000,100001) # basic wavenumber values [0:0.01:1000] (1/m)
    fA = (g*k*np.tanh(k*h))**0.5/(2*math.pi) # corresponding basic wave frequencies (Hz)
    A = np.cosh(k*(h-z))/np.cosh(k*h) # set of correction factors
    Af = np.interp(f, fA, A) # correction factor interpolated to the burst frequencies
    Ag = 0.05 # attenuation cut-off value    
    Af[Af<Ag] = np.nan        
    P1_A = P1/Af # energy density spectrum at the sea surface (m2 s)

    # ADD A HIGH FREQUENCY TAIL    
    # E(f)~f^(-4) after Kaihatu et al., 2007
    lastval = (np.isnan(P1_A)).argmax()-1 # highest frequency for which Af>Ag
    pom1 = sum(P1_A[lastval-9:lastval+1]*f[lastval-9:lastval+1]**-4)
    pom2 = sum(f[lastval-9:lastval+1]**-8)
    Amp = pom1/pom2
    P1_A[lastval+1:] = Amp*f[lastval+1:]**-4   

    # CALCULATE WAVE PARAMETERS   

    # Version 1: observed frequencies (f = 0.04:f[lastval] Hz)
    m0 = sum(P1_A[41:lastval+1]*d_f) # f[41] = 0.04 Hz
    m1 = sum(P1_A[41:lastval+1]*d_f*f[41:lastval+1])
    m2 = sum(P1_A[41:lastval+1]*d_f*f[41:lastval+1]**2)
    m_neg1 = sum(P1_A[41:lastval+1]*d_f*f[41:lastval+1]**-1)
    Hs = 4*m0**0.5 # significant wave height (m)
    Tm_01 = m0/m1 # mean absolute wave period (s)
    Tm_02 = (m0/m2)**0.5 # mean absolute zero-crossing period (s)
    Tm_neg10 = m_neg1/m0 # energy period (s)
 
    # Version 2: observed frequencies with an infinite tail (f = 0.04:inf Hz)
    m0_inf = sum(P1_A[41:lastval+1]*d_f)+Amp/3*f[lastval]**-3
    m1_inf = sum(P1_A[41:lastval+1]*d_f*f[41:lastval+1])+Amp/2*f[lastval]**-2
    m2_inf = sum(P1_A[41:lastval+1]*d_f*f[41:lastval+1]**2)+Amp*f[lastval]**-1
    m_neg1_inf = sum(P1_A[41:lastval+1]*d_f*f[41:lastval+1]**-1)+Amp/4*f[lastval]**-4
    Hs_inf = 4*m0_inf**0.5 
    Tm_01_inf = m0_inf/m1_inf 
    Tm_02_inf = (m0_inf/m2_inf)**0.5 
    Tm_neg10_inf = m_neg1_inf/m0_inf
    
    # SAVE OUTPUTS
    spectra.loc[n-1,:] = P1_A.round(6)
    properties.Burst[n-1] = n
    properties.Time[n-1] = df_time
    properties.MeanDepth[n-1] = round(h, 6)
    properties.f_lastval[n-1] = f[lastval].round(6)
    properties.Hs[n-1] = Hs.round(6)
    properties.Tm_01[n-1] = Tm_01.round(6)
    properties.Tm_02[n-1] = Tm_02.round(6)
    properties.Tm_neg10[n-1] = Tm_neg10.round(6)
    properties.Hs_inf[n-1] = Hs_inf.round(6)
    properties.Tm_01_inf[n-1] = Tm_01_inf.round(6)
    properties.Tm_02_inf[n-1] = Tm_02_inf.round(6)
    properties.Tm_neg10_inf[n-1] = Tm_neg10_inf.round(6)

spectra = spectra.drop(columns=range(41)) # drop unreliable values for f < 0.04 Hz
spectra.to_csv(spectraName, header=False, index = False)
properties.to_csv(propertiesName, header=True, index = False)
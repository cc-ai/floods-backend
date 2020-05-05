from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import pandas as pd
import netCDF4
import numpy as np

#Coded by Ata

def shift_frequency(coordinates):
    filename = 'ccai/climate/data/returnPeriodShift_HELIX_dis_rcp85_r1_statistics.nc'
    rps1 = netCDF4.Dataset(filename, "r")
    baseline_rp_shift1 = np.array(rps1.variables["baseline_rp_shift"])
    baseline_rp = np.array(rps1.variables["baseline_rp"])
    lat = np.array(rps1.variables["lat"])
    lon = np.array(rps1.variables["lon"])
    filename2 = 'ccai/climate/data/returnPeriodShift_HELIX_dis_rcp85_r2_statistics.nc'
    rps2 = netCDF4.Dataset(filename2, "r")
    baseline_rp_shift2 = np.array(rps2.variables["baseline_rp_shift"])
    filename3 = 'ccai/climate/data/returnPeriodShift_HELIX_dis_rcp85_r3_statistics.nc'
    rps3 = netCDF4.Dataset(filename3, "r")
    baseline_rp_shift3 = np.array(rps3.variables["baseline_rp_shift"])
    filename4 = 'ccai/climate/data/returnPeriodShift_HELIX_dis_rcp85_r4_statistics.nc'
    rps4 = netCDF4.Dataset(filename4, "r")
    baseline_rp_shift4 = np.array(rps4.variables["baseline_rp_shift"])
    filename5 = 'ccai/climate/data/returnPeriodShift_HELIX_dis_rcp85_r5_statistics.nc'
    rps5 = netCDF4.Dataset(filename5, "r")
    baseline_rp_shift5 = np.array(rps5.variables["baseline_rp_shift"])
    filename6 = 'ccai/climate/data/returnPeriodShift_HELIX_dis_rcp85_r6_statistics.nc'
    rps6 = netCDF4.Dataset(filename6, "r")
    baseline_rp_shift6 = np.array(rps6.variables["baseline_rp_shift"])
    filename7 = 'ccai/climate/data/returnPeriodShift_HELIX_dis_rcp85_r7_statistics.nc'
    rps7 = netCDF4.Dataset(filename7, "r")
    baseline_rp_shift7 = np.array(rps7.variables["baseline_rp_shift"])

    ####given lon by user :let say -160.27
    userlon=coordinates.lon
    findlon=lon[min(range(len(lon)), key = lambda i: abs(lon[i]-userlon))]
    indexlon=np.where(lon== findlon)

    ####given lat by user :let say 40.27
    userlat=coordinates.lat
    findlat=lat[min(range(len(lat)), key = lambda i: abs(lat[i]-userlat))]
    indexlat=np.where(lat== findlat)

    # ####baseline array([ 10.,  20.,  50., 100., 200., 500.], dtype=float32)
    #
    # ########################################## 1 in 50 years flood##########################################
    #
    # ################################ Warming Level 1.5#####################
    # #Baseline 50 ---->1995
    # warmlev=0;
    # returnperiod=2
    # #shift in frequency of flood in period of 2005-2040
    # listshifts=[baseline_rp_shift1[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift2[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift3[warmlev,returnperiod,indexlon,indexlat],
    #            baseline_rp_shift4[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift5[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift6[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift7[warmlev,returnperiod,indexlon,indexlat]]
    #
    # meanshift=np.mean(listshifts)
    # stdshift=np.std(listshifts)
    # upshift=meanshift+stdshift
    #
    # lowshift=meanshift-stdshift
    # if lowshift<0:
    #     lowshift=0

    # print([meanshift,upshift,lowshift]) # mean value of new return period, upper bound,lower bound (uncertainities)




    ################################ Warming Level 2#####################
    #Baseline 50 ---->1995
    warmlev=1
    returnperiod=2
    #shift in frequency of flood in period of 2020-2055
    listshifts=[baseline_rp_shift1[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift2[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift3[warmlev,returnperiod,indexlon,indexlat],
               baseline_rp_shift4[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift5[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift6[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift7[warmlev,returnperiod,indexlon,indexlat]]

    meanshift=np.mean(listshifts)
    stdshift=np.std(listshifts)
    upshift=meanshift+stdshift

    lowshift=meanshift-stdshift
    if lowshift<0:
        lowshift=0

    # print([meanshift,upshift,lowshift]) # mean value of new return period, upper bound,lower bound (uncertainities)

    probability = 1/(meanshift+.001)*100

    return round(probability, 1)



    ################################ Warming Level 4#####################
    #Baseline 50 ---->1995
    # warmlev=2;
    # returnperiod=2
    # #shift in frequency of flood in period of 2065-2113
    # listshifts=[baseline_rp_shift1[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift2[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift3[warmlev,returnperiod,indexlon,indexlat],
    #            baseline_rp_shift4[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift5[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift6[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift7[warmlev,returnperiod,indexlon,indexlat]]
    #
    # meanshift=np.mean(listshifts)
    # stdshift=np.std(listshifts)
    # upshift=meanshift+stdshift
    #
    # lowshift=meanshift-stdshift
    # if lowshift<0:
    #     lowshift=0
    #
    # # print([meanshift,upshift,lowshift]) # mean value of new return period, upper bound,lower bound (uncertainities)
    #
    #
    #
    # ############################## 1 in 100 years flood############################################3
    #
    #
    #
    # ################################ Warming Level 1.5#####################
    # #Baseline 100 ---->1995
    # warmlev=0;
    # returnperiod=3
    # #shift in frequency of flood in period of 2005-2040
    # listshifts=[baseline_rp_shift1[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift2[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift3[warmlev,returnperiod,indexlon,indexlat],
    #            baseline_rp_shift4[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift5[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift6[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift7[warmlev,returnperiod,indexlon,indexlat]]
    #
    # meanshift=np.mean(listshifts)
    # stdshift=np.std(listshifts)
    # upshift=meanshift+stdshift
    #
    # lowshift=meanshift-stdshift
    # if lowshift<0:
    #     lowshift=0
    #
    # # print([meanshift,upshift,lowshift]) # mean value of new return period, upper bound,lower bound (uncertainities)
    #
    #
    #
    #
    # ################################ Warming Level 2#####################
    # #Baseline 100 ---->1995
    # warmlev=1;
    # returnperiod=3
    # #shift in frequency of flood in period of 2020-2055
    # listshifts=[baseline_rp_shift1[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift2[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift3[warmlev,returnperiod,indexlon,indexlat],
    #            baseline_rp_shift4[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift5[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift6[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift7[warmlev,returnperiod,indexlon,indexlat]]
    #
    # meanshift=np.mean(listshifts)
    # stdshift=np.std(listshifts)
    # upshift=meanshift+stdshift
    #
    # lowshift=meanshift-stdshift
    # if lowshift<0:
    #     lowshift=0
    #
    # # print([meanshift,upshift,lowshift]) # mean value of new return period, upper bound,lower bound (uncertainities)
    #
    #
    #
    #
    # ################################ Warming Level 4#####################
    # #Baseline 100 ---->1995
    # warmlev=2;
    # returnperiod=3
    # #shift in frequency of flood in period of 2065-2113
    # listshifts=[baseline_rp_shift1[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift2[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift3[warmlev,returnperiod,indexlon,indexlat],
    #            baseline_rp_shift4[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift5[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift6[warmlev,returnperiod,indexlon,indexlat],baseline_rp_shift7[warmlev,returnperiod,indexlon,indexlat]]
    #
    # meanshift=np.mean(listshifts)
    # stdshift=np.std(listshifts)
    # upshift=meanshift+stdshift
    #
    # lowshift=meanshift-stdshift
    # if lowshift<0:
    #     lowshift=0
    #
    # # print([meanshift,upshift,lowshift]) # mean value of new return period, upper bound,lower bound (uncertainities)

import csv
import pandas as pd
import numpy as np

filename = 'ccai/climate/data/FloodArchive.xlsx'

def fetch_history(coordinates):
    """Looks into FloodArchive.xlsx
    fetches historical data from the lat/lon
    and returns a historical report:

    datebegin: the date that when the flood began
    floodcause: the cause of the flood
    suffered: how many people affected
    numdeath: how many people died
    dateendL how long the flood lasted
    """
    fopen = pd.read_excel(filename)

    long = fopen['long']
    lat = fopen['lat']
    country = fopen['Country']
    datebegin = fopen['Began']
    floodcause = fopen['MainCause']
    suffered = fopen['Displaced']
    numdeath = fopen['Dead']
    dateend = fopen['Ended']

    userlon = coordinates.lon
    userlat = coordinates.lat

    long_list = np.abs(long - userlon)
    lat_list = np.abs(lat - userlat)

    long_idx = np.asarray(np.where(long_list < 0.5))
    lat_idx = np.asarray(np.where(lat_list < 0.5))
    history_report = 'There is no historical flood record in your region.'
    coordinates_idx = [i for i in long_idx[0] if i in lat_idx[0]]

    if len(coordinates_idx) > 0:

        coord_idx = coordinates_idx[-1]
        floodduration = dateend[coord_idx] - datebegin[coord_idx]
        floodduration = str(floodduration).split()[0]

        history_report = 'Based on the historical archives, the most recent flood in your region has occured on ' + \
                        str(datebegin[coord_idx]).split()[0] + ' due to the ' + str(
                        floodcause[coord_idx]) + ' and lasted for ' + floodduration + ' days. ' + str(
                        suffered[coord_idx]) + ' people have sufferend from this flood and ' + str(
                        numdeath[coord_idx]) + ' people have died.'

    return history_report


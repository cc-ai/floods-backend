import math
import xarray as xr

base_dir = 'ccai/climate/data/coastal/'
file_suffix = '.tif_2050_RCP85.tif'

def fetch_coastal(coordinates):
    userlat = coordinates.lat
    userlon = coordinates.lon

    if userlat > 0:
        V_hemisphere = 'N'
    else:
        V_hemisphere = 'S'
    if userlon < 0:
        H_hemisphere = 'W'
    else:
        H_hemisphere = 'E'

    coastal_data = base_dir + V_hemisphere + str(abs(math.floor(userlat))) + H_hemisphere + str(
        abs(math.floor(userlon))).zfill(3) + file_suffix

    try:
        ds = xr.open_rasterio(coastal_data)
        coastal = ds.sel(band=1, x=userlon, y=userlat, method='nearest').values
        coastal = int(coastal * 100)
    except:
        coastal = 0

    return str(coastal)


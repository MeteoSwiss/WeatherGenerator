import xarray as xr

ds = xr.open_zarr("/capstor/store/mch/msopr/ml/datasets/mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr", consolidated=False)
print("lat:", float(ds["latitudes"].min()), float(ds["latitudes"].max()))
print("lon:", float(ds["longitudes"].min()), float(ds["longitudes"].max()))

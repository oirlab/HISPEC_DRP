import os
import requests
import tqdm
import datetime

from .. import datamodels

__all__ = ['generate_koa_filepath', 'generate_koa_id', 'download_from_koa']

login_url = "https://koa.ipac.caltech.edu/cgi-bin/KoaAPI/nph-koaLogin?"
getkoa_url = "https://koa.ipac.caltech.edu/cgi-bin/getKOA/nph-getKOA?return_mode=json&"
caliblist_url = "https://koa.ipac.caltech.edu/cgi-bin/KoaAPI/nph-getCaliblist?"


def generate_koa_filepath(input : datamodels.HISPECDataModel | str) -> str:
    """
    Returns the filepath on KOA for a given input data model. Use flat paths for now to match HISPEC_TEST_DATA.
    # KPF example = "/KPF/2024/20240926/lev0/KP.20240926.74183.78.fits"
    """
    model = datamodels.open(input, meta_only=True)
    if isinstance(model, datamodels.HISPECCalibrationModel):
        level = "cal"
    else:
        level = f"lev{model.meta.data_level}"
    date = model.meta.date_obs
    year = date[0:4]
    koa_id = generate_koa_id(model)
    return f"/HISPEC/{year}/{date}/{level}/{koa_id}"


def generate_koa_id(input : datamodels.HISPECDataModel | str) -> str:
    model = datamodels.open(input, meta_only=True)
    date = model.meta.date_obs
    utc = datetime.datetime.strptime(model.meta.utc, '%H:%M:%S.%f')
    total_seconds = utc.hour * 3600 + utc.minute * 60 + utc.second + utc.microsecond / 1e6
    time = f"{total_seconds:08.2f}"
    ext = os.path.splitext(model.meta.filename)[1]
    return f"HS.{date}.{time}{ext}"


def download_from_koa(
        koa_filename : str,
        output_dir : str,
        cookies : str | None = None
    ) -> str:

    # Make the directory
    os.makedirs(output_dir, exist_ok=True)

    # Local filename
    filename_local = os.path.join(output_dir, os.path.basename(koa_filename))

    # url
    url = getkoa_url + koa_filename

    # HTTP Request
    response = requests.get(url, stream=True, cookies=cookies)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
    else:
        # Get total file size from headers if available
        total_size = int(response.headers.get('content-length', 0))

        # Save the file with a progress bar
        with open(filename_local, 'wb') as f, tqdm.tqdm(
            total=total_size, unit='B', unit_scale=True, desc=filename_local
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Return
    return filename_local



from hispecdrp import datamodels
from astropy.time import Time
import requests
import re
import os


def add_metadata(model : datamodels.HISPECDataModel):
    """
    Adds basic metadata to a HISPEC datamodel (in-place). 

    Args:
        model (HISPECDataModel): The HISPEC datamodel to add metadata to.
    """
    # NOTE: Consider moving this method to datamodels.utils
    model.meta.instrument.name = "HISPEC"
    model.meta.file_type = 'FITS'
    model.meta.telescope = 'Keck-II'
    model.meta.organization = 'WMKO'
    model.meta.drp_version = '0.0.1'
    model.meta.instrument.era = '0.0.1'
    time = Time(model.meta.exposure.jd_start, format='jd')
    model.meta.date = time.isot
    model.meta.date_created = time.isot

    model.meta.exposure.date_start = time.datetime.strftime('%Y%m%d')
    model.meta.exposure.date_mid = time.datetime.strftime('%Y%m%d')
    model.meta.exposure.date_end = time.datetime.strftime('%Y%m%d')
    model.meta.exposure.time_start = time.datetime.strftime('%H:%M:%S.%f')[:-3]
    model.meta.exposure.time_mid = time.datetime.strftime('%H:%M:%S.%f')[:-3]
    model.meta.exposure.time_end = time.datetime.strftime('%H:%M:%S.%f')[:-3]


def download_github_lfs_file(
        git_file_path: str,
        output_dir: str | None = None,
        preserve_github_path: bool = True,
        base_url: str | None = None,
        use_cached : bool = True
    ) -> str | None:
    """
    Download a file from HISPEC_Test_Data.
    See: https://gist.github.com/fkraeutli/66fa741d9a8c2a6a238a01d17ed0edc5#retrieving-lfs-files
    
    Args:
        git_file_path (str): Path to the file in the HISPEC_Test_Data repository.
        output_dir (str, optional): Local directory to save the file (default: current directory).
        preserve_github_path (bool, optional): Preserve the GitHub path in the output directory (default: True)
        base_url (str, optional): Base URL of the HISPEC_Test_Data repository
            Defaults to: "https://raw.githubusercontent.com/oirlab/HISPEC_Test_Data/main/"
        use_cached (bool, optional): Use cached file if it exists (default: True).
    
    Returns:
        str | None: Path to the downloaded file or None if download failed.
    """

    if base_url is None:
        base_url = "https://raw.githubusercontent.com/oirlab/HISPEC_Test_Data/main/"
    
    # LFS URL
    lfs_batch_url = "https://github.com/oirlab/HISPEC_Test_Data.git/info/lfs/objects/batch"
    
    # Set the output path
    if preserve_github_path:
        output_path = os.path.join(output_dir, git_file_path)
    else:
        output_path = os.path.join(output_dir, os.path.basename(git_file_path))

    if os.path.exists(output_path) and use_cached:
        return output_path
    
    # Ensure the output directory exists
    if output_dir:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Get the file URL from GitHub's raw content
        url = base_url + git_file_path
        response = requests.get(url)
        response.raise_for_status()

        # Check if the file is an LFS pointer (starts with 'version')
        if response.text.startswith('version'):
            # Extract the oid (SHA-256 hash) and size from the LFS pointer file
            match = re.search(r'oid sha256:([a-f0-9]{64})', response.text)
            size_match = re.search(r'size (\d+)', response.text)
            if not match or not size_match:
                raise ValueError(f"Unable to find 'oid' or 'size' in the LFS pointer file: {response.text}")

            lfs_oid = match.group(1)
            lfs_size = size_match.group(1)

            # Build the JSON payload for LFS batch API
            payload = {
                "operation": "download",
                "transfer": ["basic"],
                "objects": [{"oid": lfs_oid, "size": int(lfs_size)}]
            }

            # API URL to interact with LFS objects batch download
            headers = {
                'Accept': 'application/vnd.git-lfs+json',
                'Content-Type': 'application/json',
            }

            # Send the request to GitHub LFS API
            lfs_response = requests.post(lfs_batch_url, headers=headers, json=payload)
            lfs_response.raise_for_status()

            # Extract the URL for the LFS object
            lfs_data = lfs_response.json()
            download_url = lfs_data['objects'][0]['actions']['download']['href']

            # Download the file from the href URL
            file_data = requests.get(download_url).content

        else:
            file_data = response.content

        # Write the file content to the output path
        with open(output_path, 'wb') as f:
            f.write(file_data)
        
        print(f"Successfully downloaded {url} to {output_path}")
        return output_path

    except Exception as e:
        print(f"Error downloading file at {url}:\n{e}")
        return None
    





# PREDEFINED_DATA_FOLDERS=["data/"]

# for folder in PREDEFINED_DATA_FOLDERS:
#     full_path = os.path.join(folder, filename)
#     if os.path.exists(full_path):
#         warnings.warn(f"Access data from {full_path}")
#         return full_path
#     with data.conf.set_temp("dataurl", DATAURL), data.conf.set_temp(
#         "remote_timeout", 30
#     ):
#         warnings.warn(f"Retrieve data for {filename} (if not cached already)")
#         local_path = data.get_pkg_data_filename(filename, show_progress=True)
#         local_path_suffix = local_path + ".fits"
#         if not os.path.exists(local_path_suffix):
#             os.symlink(local_path, local_path_suffix)
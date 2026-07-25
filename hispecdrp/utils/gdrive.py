import os
import gdown
import astropy.utils.data

__all__ = ['download_gdrive_file']

from .testing_data import GDRIVE_FILES

def get_id_from_path(filepath: str) -> str:

    # Full relative filepath
    if filepath in GDRIVE_FILES:
        return GDRIVE_FILES[filepath]
    
    # Filename only
    for key in GDRIVE_FILES:
        if key.endswith(filepath):
            return GDRIVE_FILES[key]

    # Error
    raise ValueError(f"Filepath '{filepath}' not found in GDRIVE_FILES mapping.")

def download_gdrive_file(
    filepath: str,
    output_dir: str | None = None,
    use_cached: bool = True
) -> str | None:
    
    file_id = get_id_from_path(filepath)
    if output_dir is None:
        output_dir = os.getenv('HISPECDRP_TEST_DATA_DIR')
        if output_dir is None:
            output_dir = os.path.join(
                str(astropy.utils.data._get_download_cache_loc()),
                'HISPECDRP_TEST_DATA'
            )
    
    download_url = f"https://drive.google.com/uc?id={file_id}"

    output_path = os.path.join(output_dir, os.path.basename(filepath))
    
    if os.path.exists(output_path) and use_cached:
        return output_path
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading {filepath} with id {file_id} to {output_path}")

    gdown.download(download_url, output_path, quiet=False)

    return output_path
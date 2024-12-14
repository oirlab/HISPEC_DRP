import os
from astropy.io import fits
import pandas as pd
import numpy as np
from typing import Type
import datetime
import git

class DataModel:

    extensions = {
        'PRIMARY': fits.PrimaryHDU,
        'RECEIPT': fits.BinTableHDU,
    }

    RECEIPT_COLS = [
        'Time',
        'Code_Release',
        'Branch_Name',
        'Commit_Hash',
        'Class',
        'Datamodel',
        'Arguments',
        'Status'
    ]

    HDU_TYPE_MAP = {
        fits.ImageHDU: np.array,
        fits.BinTableHDU: pd.DataFrame
    }


    def __init__(self, filename : str | None, extensions : str | list[str] | None = None):
        self.filename = filename
        self.read_headers(extensions=extensions)
        self.read_data(extensions=extensions)
        self.receipt = pd.DataFrame()


    @property
    def dirname(self):
        return os.path.dirname(self)


    @property
    def basename(self):
        return os.path.basename(self.filename)

    @property
    def file_ext(self):
        return os.path.splitext(self.filename)[1]
    
    @property
    def extnames(self):
        return list(self.extensions.keys())


    def read_headers(self) -> dict[str, fits.Header]:
        self.headers = {}
        with fits.open(self.filename, memmap=True) as hdulist:
            for extname in self.extensions:
                self.headers[extname] = hdulist[extname].header
        return self.headers
    

    def read_data(self, extensions : str | list[str] | None = None) -> dict[str, fits.hdu.base._ValidHDU]:
        if extensions is None:
            extensions = list(self.extensions.keys())
        if isinstance(extensions, str):
            extensions = [extensions]
        extensions = [e.upper() for e in extensions]
        self.data = {}
        with fits.open(self.filename, memmap=True) as hdulist:
            for hdu in hdulist:
                if hdu.name in extensions:
                    hdu = hdulist[hdu.name]
                    data = hdu.data
                    init = self.HDU_TYPE_MAP[type(data)]
                    self.data[hdu.name] = init(data)
        return self.data


    @classmethod
    def from_file(cls, filename : str, extensions : str | list[str] | None = None):
        return cls(filename, extensions=extensions)
    

    def save(self, filename_out : str | None):
        if filename_out is None:
            filename_out = self.filename
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.headers['PRIMARY']))
        for extname, data in self.data.values():
            init = self.extensions[extname]
            hdu = init(data=data, name=extname, header=self.headers[extname])
            hdulist.append(hdu)
        hdulist.writeto(filename_out)


    def get_calibration_file(self, kind : str, **kwargs):
        pass

    def add_receipt_entry(self, primitive, status : str | None):
        """
        Taken from KPF DRP and modified:
            https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/master/kpfpipe/models/base_model.py
        """
        time = datetime.datetime.now().isoformat()

        # Get version info
        repo = git.Repo(search_parent_directories=True)
        git_commit_hash = repo.head.object.hexsha
        git_branch = repo.active_branch.name
        git_tag = str(repo.tags[-1])

        # add the row to the bottom of the table
        row = {
            'Time' : time,
            'Code_Release': git_tag,
            'Branch_Name': git_branch,
            'Commit_Hash': git_commit_hash,
            'Class': f"{primitive.__class__.__module__}.{primitive.__class__.__name__}",
            'Filename' : self.filename,
            'Arguments': repr(primitive.args),
            'Status' : status
        }
        self.receipt.append(row, ignore_index=True)
        
        # Add DRPTAG and DRPHASH to primary header
        # NOTE: MOVE THIS SOMEWHERE ELSE
        self.header['PRIMARY']['DRPTAG'] = git_tag
        self.header['PRIMARY']['DRPHASH'] = git_commit_hash


    def __repr__(self):
        return f"{self.__class__.__name__}('{self.filename}')"
    

    def __init_subclass__(cls):
        subclasses = cls.mro()[::-1]
        for _cls in subclasses:
            if hasattr(_cls, "extensions"):
                cls.extensions.update(_cls.extensions)
            if hasattr(_cls, "HDU_TYPE_MAP"):
                cls.HDU_TYPE_MAP.update(_cls.HDU_TYPE_MAP)
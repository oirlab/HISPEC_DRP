from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="hispecdrp.primitives.ramp_fitting.fitramp_cython_Brandt",
        sources=["hispecdrp/primitives/ramp_fitting/fitramp_cython_Brandt.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
    )
)
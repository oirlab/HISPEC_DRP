from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="hispecdrp.primitives.ramp_fitting.fitramp_cython_Brandt",
        sources=["hispecdrp/primitives/ramp_fitting/fitramp_cython_Brandt.pyx"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
    )
)
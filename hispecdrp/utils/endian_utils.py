import numpy as np
import sys

__all__ = ['convert_endianness']

def convert_endianness(arr : np.ndarray, output_endianness : str = 'little'):
    if arr.dtype.byteorder == '=':
        current_endianness = sys.byteorder
    elif arr.dtype.byteorder == '<':
        current_endianness = 'little'
    else:   
        current_endianness = 'big'
    if current_endianness != output_endianness:
        return arr.byteswap().view(arr.dtype.newbyteorder(output_endianness))
    else:
        return arr
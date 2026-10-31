"""Validated one-slot Windows mapping; return owned pixels before next request."""
import ctypes
import mmap
import re
import struct
import sys
import numpy as np
from rs_v612_gameplay_guard import GuardUncertainError


def open_frame_mapping(name,size):
    if sys.platform!='win32':raise GuardUncertainError('Shared frame mapping requires Windows')
    if not re.fullmatch(r'Local\\RSFrame_[0-9a-f]{48}',name or ''):
        raise GuardUncertainError('Invalid shared frame mapping name')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenFileMappingW.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_wchar_p]
    kernel.OpenFileMappingW.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenFileMappingW(4,False,name)
    if not handle:raise GuardUncertainError('Shared frame mapping does not exist')
    try:return mmap.mmap(-1,size,tagname=name,access=mmap.ACCESS_READ)
    finally:kernel.CloseHandle(handle)


def read_shared_frame(mapping,metadata,width,height):
    before=mapping[:32]
    expected=(0x52534631,1,width,height,metadata['serial'],width*height*4)
    if len(before)!=32 or struct.unpack('<IIIIQQ',before)!=expected:
        raise GuardUncertainError('Shared frame header/serial/geometry mismatch')
    # Retain the existing capture API's ownership contract. This ~7MB copy
    # replaces socket transfer; callers can safely retain old frames afterward.
    pixels=np.frombuffer(mapping,np.uint8,count=width*height*4,offset=64).copy()
    if mapping[:32]!=before:
        raise GuardUncertainError('Shared frame overwritten during read')
    return pixels

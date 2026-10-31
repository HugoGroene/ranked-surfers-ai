"""Owned shared CPU arrays. The owner unlinks only after all clients stop."""
from multiprocessing import shared_memory
import math
import numpy as np


class SharedArrays:
    def __init__(self, specification=None, *, descriptor=None):
        self.owner = descriptor is None
        self.closed = False
        if self.owner:
            offset = 64
            layouts = {}
            for key, (shape, dtype) in specification.items():
                dtype = np.dtype(dtype)
                size = math.prod(shape) * dtype.itemsize
                offset = (offset + 63) // 64 * 64
                layouts[key] = dict(shape=list(shape), dtype=dtype.str, offset=offset)
                offset += size
            self.memory = shared_memory.SharedMemory(create=True, size=offset)
            self.descriptor = dict(name=self.memory.name, size=offset, layouts=layouts)
        else:
            self.descriptor = descriptor
            self.memory = shared_memory.SharedMemory(name=descriptor['name'])
            # Windows reports a page-rounded allocation when attaching.
            if self.memory.size < descriptor['size']:
                self.memory.close()
                raise ValueError('Shared array allocation size mismatch')
        self.arrays = {}
        for key, layout in self.descriptor['layouts'].items():
            shape = tuple(layout['shape'])
            dtype = np.dtype(layout['dtype'])
            end = layout['offset'] + math.prod(shape) * dtype.itemsize
            if dtype.hasobject or any(n <= 0 for n in shape) or layout['offset'] < 64 or end > self.descriptor['size']:
                raise ValueError('Invalid shared array layout')
            self.arrays[key] = np.ndarray(shape, dtype, buffer=self.memory.buf, offset=layout['offset'])

    def close(self):
        if self.closed:
            return
        self.arrays.clear()
        self.memory.close()
        if self.owner:
            self.memory.unlink()
        self.closed = True

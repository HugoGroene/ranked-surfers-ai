"""One published rollout slot per worker; no writes before optimizer commit."""
import struct
import numpy as np
from tools.ppo_shared_arrays import SharedArrays
from tools.ppo_parallel_learning import BUFFER_FIELDS
from tools.ppo_compact_buffer import CompactImages


class RolloutSlot:
    def __init__(self, buffer=None, *, descriptor=None):
        if descriptor is None:
            compact = isinstance(buffer.observations['image'], CompactImages)
            specification = {}
            for key, value in buffer.observations.items():
                if key == 'image' and compact:
                    specification['frames'] = ((value.capacity,3,*value.shape[-2:]), value.dtype)
                    specification['refs'] = (value.refs.shape, np.int32)
                else:
                    specification['obs_'+key] = (value.shape, value.dtype)
            for key in BUFFER_FIELDS:
                value = getattr(buffer, key)
                specification[key] = (value.shape, value.dtype)
            self.shared = SharedArrays(specification)
            self.descriptor = dict(arrays=self.shared.descriptor, compact=compact,
                                   image_shape=list(buffer.observations['image'].shape))
            struct.pack_into('<q', self.shared.memory.buf, 0, -1)
        else:
            self.descriptor = descriptor
            self.shared = SharedArrays(descriptor=descriptor['arrays'])
        self.published = None
        self.next_version=0

    def publish(self, buffer, worker, version, digest):
        if self.published is not None or version!=self.next_version or not buffer.full or buffer.generator_ready:
            raise ValueError('Rollout slot unavailable or incomplete')
        struct.pack_into('<q', self.shared.memory.buf, 0, -1)
        arrays = self.shared.arrays
        image = buffer.observations['image']
        count = None
        for key, value in buffer.observations.items():
            if key == 'image' and self.descriptor['compact']:
                count = image.count
                for i in range(count):arrays['frames'][i] = image.frames[i]
                arrays['refs'][:] = image.refs
            else:
                arrays['obs_'+key][:] = value
        for key in BUFFER_FIELDS:
            arrays[key][:] = getattr(buffer, key)
        struct.pack_into('<q', self.shared.memory.buf, 0, version)
        self.published = version
        return dict(worker=worker, version=version, policy_hash=digest,
                    shared_slot=self.shared.memory.name, frame_count=count)

    def packet(self, message, version, digest):
        if (message['version'] != version or message['policy_hash'] != digest or
                message['shared_slot'] != self.shared.memory.name or
                struct.unpack_from('<q', self.shared.memory.buf)[0] != version):
            raise ValueError('Unpublished or stale rollout slot')
        arrays = self.shared.arrays
        for value in arrays.values():value.flags.writeable=False
        observations = {key[4:]: value for key,value in arrays.items() if key.startswith('obs_')}
        if self.descriptor['compact']:
            count = message['frame_count']
            if not isinstance(count, int) or not 0 < count <= len(arrays['frames']):
                raise ValueError('Invalid published frame count')
            image = CompactImages(self.descriptor['image_shape'], frames=arrays['frames'], refs=arrays['refs'])
            image.count = count
            if (image.refs < 0).any() or (image.refs >= count).any():
                raise ValueError('Invalid published frame reference')
            observations['image'] = image
        return dict(worker=message['worker'], version=version, policy_hash=digest,
                    observations=observations, **{key:arrays[key] for key in BUFFER_FIELDS})

    def acknowledge(self, version):
        if self.published != version:
            raise ValueError('Wrong rollout commit version')
        self.published = None
        self.next_version+=1

    def close(self):
        self.shared.close()

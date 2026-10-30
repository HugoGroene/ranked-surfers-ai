"""Inspect, or explicitly set ONLY non-primary MTT virtual monitors to 1080p144.

Uses the Win32 DEVMODEW ABI (220 bytes); never stages a physical display change.
Windows may reset virtual modes when its driver is reloaded, so inspect on enable.
"""
import argparse
import ctypes as c
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--refresh-hz', type=int, choices=[30, 60, 144], default=144,
                        help='Virtual monitors only; default preserves 144 Hz')
    args = parser.parse_args(argv)
    if sys.platform != 'win32':
        parser.error('Windows display configuration only')
    from ctypes import wintypes as w
    class Device(c.Structure):
        _fields_ = [('size', w.DWORD), ('name', w.WCHAR*32), ('description', w.WCHAR*128),
                    ('flags', w.DWORD), ('id', w.WCHAR*128), ('key', w.WCHAR*128)]
    api = c.WinDLL('user32', use_last_error=True)
    api.EnumDisplayDevicesW.argtypes = [w.LPCWSTR, w.DWORD, c.POINTER(Device), w.DWORD]
    api.EnumDisplaySettingsW.argtypes = [w.LPCWSTR, w.DWORD, c.c_void_p]
    api.ChangeDisplaySettingsExW.argtypes = [w.LPCWSTR, c.c_void_p, w.HWND, w.DWORD, c.c_void_p]
    def mode(name, index=0xFFFFFFFF):
        value = c.create_string_buffer(220)
        c.c_ushort.from_buffer(value, 68).value = 220
        return value if api.EnumDisplaySettingsW(name, index, value) else None
    def describe(value):
        return {key: c.c_int32.from_buffer(value, offset).value for key, offset in
                [('left',76), ('top',80), ('width',172), ('height',176), ('refresh_hz',184)]}
    displays = []
    for index in range(64):
        device = Device(size=c.sizeof(Device))
        if not api.EnumDisplayDevicesW(None, index, c.byref(device), 0):
            break
        if not device.flags & 1:
            continue
        child = Device(size=c.sizeof(Device))
        api.EnumDisplayDevicesW(device.name, 0, c.byref(child), 0)
        value = mode(device.name)
        if value is None:
            raise RuntimeError('Cannot read active display settings; no changes allowed')
        displays.append(dict(name=device.name, adapter=device.description,
            virtual='MTT1337' in child.id.upper(), primary=bool(device.flags & 4), mode=value))
    virtual = [d for d in displays if d['virtual']]
    physical = [d for d in displays if not d['virtual']]
    if not 1 <= len(virtual) <= 2 or not physical or any(d['primary'] for d in virtual):
        raise RuntimeError('Expected one or two non-primary MTT displays plus a physical monitor')
    before = {d['name']: describe(d['mode']) for d in displays}
    print(json.dumps(dict(before=before, applied=False), indent=2))
    if not args.apply:
        return
    x = max(describe(d['mode'])['left']+describe(d['mode'])['width'] for d in physical)
    staged = []
    for display in virtual:
        supported = False
        for index in range(4096):
            value = mode(display['name'], index)
            if value is None:
                break
            info = describe(value)
            if (info['width'], info['height'], info['refresh_hz']) == (1920,1080,args.refresh_hz):
                supported = True
                break
        if not supported:
            raise RuntimeError(f'VDD does not advertise 1920x1080 at {args.refresh_hz} Hz; no changes staged')
        value = c.create_string_buffer(display['mode'].raw, 220)
        # DM_POSITION|DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT|DM_DISPLAYFREQUENCY
        for offset, number in [(72,0x5c0020),(76,x),(80,0),(168,32),(172,1920),(176,1080),(184,args.refresh_hz)]:
            c.c_int32.from_buffer(value, offset).value = number
        if api.ChangeDisplaySettingsExW(display['name'], value, None, 2, None) != 0:
            raise RuntimeError('Virtual mode test failed; no changes staged')
        staged.append((display['name'], value))
        x += 1920
    try:
        for name, value in staged:
            if api.ChangeDisplaySettingsExW(name, value, None, 0x10000001, None) != 0:
                raise RuntimeError('Virtual mode staging failed')
        if api.ChangeDisplaySettingsExW(None, None, None, 0, None) != 0:
            raise RuntimeError('Virtual mode application failed')
    except BaseException:
        for display in virtual:
            api.ChangeDisplaySettingsExW(display['name'], display['mode'], None, 0x10000001, None)
        api.ChangeDisplaySettingsExW(None, None, None, 0, None)
        raise
    after = {d['name']: describe(mode(d['name'])) for d in displays}
    for display in physical:
        if after[display['name']] != before[display['name']]:
            raise RuntimeError('Physical mode changed unexpectedly; inspect Windows display settings')
    print(json.dumps(dict(after=after, applied=True, physical_modes_unchanged=True), indent=2))


if __name__ == '__main__':
    main()

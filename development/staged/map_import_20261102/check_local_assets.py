"""Read-only dependency preflight for the saved new map; does not launch gameplay."""
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urljoin

ROOT=Path(__file__).resolve().parents[3]
MAP=ROOT/'rankedsurfers-maps/current_20261102'

def inspect():
    entry=json.loads((MAP/'actual_entry.json').read_text())['gameEntry']
    missing=set()
    for file in MAP.rglob('*'):
        if not file.is_file() or file.suffix not in ('.html','.js','.mjs','.css','.json'):
            continue
        content=file.read_text(encoding='utf-8',errors='replace')
        for path in re.findall(r'''["']((?:/|\./|\.\./)[^"'\s<>]+)["']''',content):
            path=urljoin('/'+file.relative_to(MAP).as_posix(),path)
            path=urlsplit(path).path
            if not path.endswith(('.js','.mjs','.css','.wasm','.unityweb','.woff2')):
                continue
            target=(MAP/path.lstrip('/')).resolve()
            if target.is_relative_to(MAP.resolve()) and not target.is_file():missing.add(path)
    manifest=MAP/entry.lstrip('/')
    manifest=manifest.with_name('manifest.json')
    assets=json.loads(manifest.read_text()) if manifest.exists() else {}
    binaries={k:dict(path=v,exists=(MAP/v.lstrip('/')).is_file(),
                    bytes=(MAP/v.lstrip('/')).stat().st_size if (MAP/v.lstrip('/')).is_file() else 0)
              for k,v in assets.items()}
    return dict(status='INCOMPLETE_ASSETS' if missing or not assets else 'STATIC_PREFLIGHT_ONLY',
                map_entry=entry,assets=binaries,missing_static_references=sorted(missing),
                live_checks={k:'NOT_RUN' for k in ('startup','gameplay','death','pause_resume','crop','timing','input_effectiveness')},
                note='Static references are conservative; dynamic dependencies and browser behavior still require validation.')

if __name__=='__main__':
    result=inspect()
    dest=ROOT/'reports/new_map_preflight_20261104.json'
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('REPORT:',dest)

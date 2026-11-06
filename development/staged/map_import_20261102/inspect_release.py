"""Download public HTML/release metadata, never execute the remote game."""
import json
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error
import gzip
import re

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'rankedsurfers-maps/current_20261102'
BASE='https://rankedsurfers.com'
HEADERS={'User-Agent':'Mozilla/5.0', 'Accept-Encoding':'identity'}

def fetch(path):
    url=urllib.parse.urljoin(BASE,path)
    if urllib.parse.urlsplit(url).hostname!='rankedsurfers.com':raise ValueError(url)
    dest=OUT/urllib.parse.urlsplit(url).path.lstrip('/') if urllib.parse.urlsplit(url).path!='/' else OUT/'index.html'
    if dest.is_file():return dest.read_text(encoding='utf-8'),dest
    with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=60) as r:
        data=r.read()
        if r.headers.get('Content-Encoding')=='gzip' and data[:2]==b'\x1f\x8b':data=gzip.decompress(data)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_bytes()!=data:raise FileExistsError(dest)
    if not dest.exists():dest.write_bytes(data)
    return data.decode('utf-8'),dest

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    home,_=fetch('/')
    print('HOME ASSETS:',re.findall(r'(?:src|href)=[\"\x27]([^\"\x27]+)',home))
    release,_=fetch('/release.json')
    print('RELEASE:',release)
    entry=json.loads(release)['gameEntry']
    try:
        arena,path=fetch(entry)
        print('ARENA:',str(path),'bytes:',len(arena))
    except urllib.error.HTTPError as exc:
        print('Manifest game entry HTTP:',exc.code)
    for asset in re.findall(r'(?:src|href)=[\"\x27]([^\"\x27]+)',home):
        if asset.startswith('/assets/') and '.js' in asset:
            text,path=fetch(asset)
            print('Saved JS:',path.name,len(text))
            for candidate in sorted(set(re.findall(r'/game/releases/[a-z0-9]+/arena\.html',text))):
                arena,arena_path=fetch(candidate)
                print('Current game entry:',candidate,len(arena))
                (OUT/'actual_entry.json').write_text(json.dumps({'gameEntry':candidate},indent=2))
    (OUT/'inspection.json').write_text(json.dumps({'release':json.loads(release),'arena_path':entry},indent=2))

"""Copy the authorised original classifier dataset without modifying its source."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path.home()/'Documents/ranked-surfers-ai/ai_data/v4_screen_states_rebuild_01'
DATA=ROOT/'ai_data/multimap_screen_states'

def main():
    target=DATA/'archives'/datetime.now().strftime('original_rebuild01_%Y%m%d_%H%M%S_%f')
    target.mkdir(parents=True,exist_ok=False)
    records=[]
    for src in sorted(SOURCE.rglob('*')):
        if not src.is_file():continue
        if src.is_symlink():raise ValueError('Unexpected symlink')
        relative=src.relative_to(SOURCE)
        dest=target/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest)
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:raise RuntimeError('Copy mismatch')
        records.append(dict(path=str(relative),bytes=dest.stat().st_size,sha256=digest))
    manifest=dict(source=str(SOURCE),map_id='original',status='ARCHIVED_REVIEW_REQUIRED',
        previous_validation_sessions=['rebuild01_g'],files=records,
        note='Legacy desktop geometry and labels need review; not automatically included in training.')
    (target/'archive_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('VERIFIED ARCHIVE:',target)
    print('FILES:',len(records),'BYTES:',sum(r['bytes'] for r in records))

if __name__=='__main__':main()

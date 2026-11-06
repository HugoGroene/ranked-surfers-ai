"""CPU-only export helpers: reviewed-state provenance and compact frame storage."""
import csv
import hashlib
from pathlib import Path
import numpy as np
from common import read,digest


def reviewed_states(root):
    result={};provenance=[]
    if root is None:return result,provenance
    for file in sorted((Path(root)/'sessions').glob('*/session.json')):
        meta=read(file);labels=file.parent/'labels.csv';manifest=file.parent/'import_manifest.json'
        if not meta.get('bc_source_session') or not labels.exists():continue
        source=Path(meta['bc_source_session']).resolve()
        frames={row['file']:row for row in read(manifest)['frames']}
        provenance.append(dict(session=str(file.parent.resolve()),split=meta['split'],
            labels_sha256=digest(labels),import_sha256=digest(manifest)))
        with labels.open(newline='',encoding='utf8') as stream:
            for label in csv.DictReader(stream):
                if label['label'] not in ('MENU','LOADING','GAMEPLAY','OTHER'):raise ValueError('Invalid reviewed state')
                row=frames[label['filename']];path=Path(row['source']).resolve()
                if not path.is_relative_to(source):raise ValueError('Reviewed frame outside source session')
                if digest(file.parent/row['file'])!=row['sha256']:raise ValueError('Reviewed guard image changed')
                entry=dict(state=label['label'],sha256=row['source_sha256'],split=meta['split'])
                if str(path) in result and result[str(path)]!=entry:raise ValueError('Conflicting reviewed states')
                result[str(path)]=entry
    return result,provenance


def apply_reviewed(rows,run,split,reviewed):
    effective=[];counts={}
    for row in rows:
        value=reviewed.get(str((run/'frames'/row['file']).resolve()))
        if value:
            if value['sha256']!=row['sha256'] or value['split']!=split:
                raise ValueError('Reviewed state checksum/split mismatch')
            if row['state']!=value['state']:
                key=row['state']+'->'+value['state'];counts[key]=counts.get(key,0)+1
            row=dict(row,state=value['state'])
        effective.append(row)
    return effective,counts


def compact_indices(samples):
    original=np.stack([s['indices'] for s in samples]).astype(np.int64)
    used=np.unique(original[original>=0])
    compact=original.copy();valid=original>=0
    compact[valid]=np.searchsorted(used,original[valid])
    return used,compact

"""Read only TAR headers and one walking member using an authenticated session.

The caller supplies a requests.Session in memory. No credentials/cookies are saved.
This deliberately requires HTTP 206: a server ignoring Range must not download 2.7GB.
"""
import csv
import io
import json
from pathlib import Path
import tarfile
import time


def fetch(session, output, max_headers=150):
    output = Path(output)
    base = 'https://download.is.tue.mpg.de/download.php?domain=bedlam2&sfile=ground_truth/motions/'
    url = base + 'b2_motions_npz_training.tar'
    referer = 'https://bedlam2.is.tuebingen.mpg.de/download.php'
    rows = list(csv.DictReader((output/'b2_bedlam2amass_motion_training.csv').open()))
    mapping = {r['body_id']+'_'+r['motion_id']+'.npz': r for r in rows}
    requests_log = []
    started = time.monotonic()

    def read_range(offset, size):
        with session.get(url, headers={'Referer': referer,
                         'Range': f'bytes={offset}-{offset+size-1}'},
                         timeout=40, stream=True) as response:
            if response.status_code != 206:
                raise RuntimeError(f'Range not honored: {response.status_code}')
            expected = f'bytes {offset}-{offset+size-1}/'
            if not response.headers.get('Content-Range', '').startswith(expected):
                raise RuntimeError('Unexpected Content-Range')
            data = response.content
            if len(data) != size:
                raise RuntimeError('Truncated range')
            requests_log.append(dict(offset=offset, bytes=size))
            return data

    offset = 0
    inspected = []
    for _ in range(max_headers):
        header = tarfile.TarInfo.frombuf(read_range(offset, 512), 'utf-8', 'strict')
        row = mapping.get(Path(header.name).name)
        inspected.append(dict(name=header.name, size=header.size, offset=offset, source=row))
        if row:
            name = Path(row['motion_rel_fname']).name.lower()
            simple_walk = name.startswith('walk_') or name.startswith('walking')
            print(header.name, row['ds_name'], row['motion_rel_fname'], flush=True)
            if simple_walk and not (row['ds_name']=='KIT' and row['motion_rel_fname'].startswith('314/')):
                if header.size > 5_000_000:
                    raise RuntimeError('Single-member size exceeds minimal acquisition guard')
                target = output/Path(header.name).name
                if target.exists():
                    raise FileExistsError(target)
                target.write_bytes(read_range(offset+512, header.size))
                record = dict(status='downloaded_unreviewed', source=row, member=header.name,
                    local_file=str(target.resolve()), member_bytes=header.size, archive_url=url,
                    archive_bytes=2863616000, requests=requests_log, inspected=inspected,
                    transferred_tar_bytes=sum(r['bytes'] for r in requests_log),
                    mapping_bytes=(output/'b2_bedlam2amass_motion_training.csv').stat().st_size,
                    license_url='https://bedlam2.is.tuebingen.mpg.de/license.html',
                    seconds=time.monotonic()-started,
                    next_gate='locked-head parameter convention and actual mesh review; not accepted by filename')
                (output/'acquisition.json').write_text(json.dumps(record, indent=2)+'\n')
                return record
        offset += 512 + ((header.size+511)//512)*512
    (output/'header_search.json').write_text(json.dumps(inspected, indent=2)+'\n')
    raise RuntimeError('No simple walking filename in bounded header search')

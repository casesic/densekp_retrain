"""Bounded, CPU-only audit of Zenodo 20800829; no reconstruction or inference.

Reads the already acquired XML and ZIP tail. HTTP range requests access the same
public archive only. Existing outputs are reused, never replaced. CRC32 verifies
ZIP members; no cryptographic hashes are computed.
"""
from pathlib import Path, PurePosixPath
import argparse
import collections
import concurrent.futures
import io
import json
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile
import zlib

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/audits/mtb_cycling_pose_dataset_a2_20261009'
URL = 'https://zenodo.org/api/records/20800829/files/images.zip/content'
TAIL_OFFSET = 665049006
ARCHIVE_SIZE = 665114542


def save(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def index():
    with zipfile.ZipFile(io.BytesIO((OUT / 'zip_tail.bin').read_bytes())) as z:
        entries = z.infolist()
    items = []
    for info in entries:
        path = PurePosixPath(info.filename)
        assert not path.is_absolute() and '..' not in path.parts
        assert (info.external_attr >> 16) & 0o170000 != 0o120000
        items.append(dict(name=info.filename, offset=info.header_offset + TAIL_OFFSET,
                          compressed_size=info.compress_size, size=info.file_size,
                          crc32=info.CRC, method=info.compress_type))
    items.sort(key=lambda x: x['offset'])
    for a, b in zip(items, items[1:]):
        a['end'] = b['offset']
    items[-1]['end'] = ARCHIVE_SIZE - len((OUT / 'zip_tail.bin').read_bytes())
    # Last member may overlap tail: size+safe local-header allowance is sufficient.
    items[-1]['end'] = min(ARCHIVE_SIZE, items[-1]['offset'] + items[-1]['compressed_size'] + 512)
    return items


def fetch(item):
    name = PurePosixPath(item['name']).name
    target = OUT / 'images' / name
    if target.exists():
        return dict(name=item['name'], reused=True, downloaded_bytes=0)
    start, end = item['offset'], item['end']
    assert 0 <= start < end <= ARCHIVE_SIZE and end - start < 12000000
    prefix = OUT / 'images.partial.zip'
    t = time.perf_counter()
    if prefix.exists() and end <= prefix.stat().st_size:
        with prefix.open('rb') as f:
            f.seek(start)
            data = f.read(end - start)
        transferred = 0
        origin = 'retained_partial_archive'
    else:
        log = OUT / 'range_logs' / name
        log.parent.mkdir(exist_ok=True)
        cmd = ['curl', '-L', '--fail', '--silent', '--show-error', '--max-time', '180',
               '--max-filesize', str(end-start), '--range', f'{start}-{end-1}',
               '-D', str(log.with_suffix('.headers.txt')), URL]
        result = subprocess.run(cmd, capture_output=True)
        data = result.stdout
        log.with_suffix('.stderr.txt').write_bytes(result.stderr)
        if result.returncode:
            log.with_suffix('.failed.bin').write_bytes(data)
            raise RuntimeError(f'{name}: curl {result.returncode}; retained {len(data)} bytes')
        headers = log.with_suffix('.headers.txt').read_text().lower()
        assert f'content-range: bytes {start}-{end-1}/{ARCHIVE_SIZE}' in headers
        assert len(data) == end-start
        transferred = len(data)
        origin = 'official_http_range_206'
    assert data[:4] == b'PK\x03\x04'
    _, _, flags, method, _, _, _, _, _, nlen, elen = struct.unpack('<IHHHHHIIIHH', data[:30])
    assert not flags & 1
    filename = data[30:30+nlen].decode('utf8')
    assert filename == item['name']
    compressed = data[30+nlen+elen:30+nlen+elen+item['compressed_size']]
    raw = zlib.decompress(compressed, -15) if method == 8 else compressed
    assert len(raw) == item['size'] and zlib.crc32(raw) == item['crc32']
    im = Image.open(io.BytesIO(raw))
    im.verify()
    target.parent.mkdir(exist_ok=True)
    with target.open('xb') as f:
        f.write(raw)
    return dict(name=item['name'], origin=origin, downloaded_bytes=transferred,
                extracted_bytes=len(raw), range=[start,end], seconds=time.perf_counter()-t,
                zip_crc32_verified=True)


def annotations():
    root = ET.parse(OUT / 'annotations.xml').getroot()
    tasks = []
    for ordinal, task in enumerate(root.findall('./meta/project/tasks/task')):
        tid = task.findtext('id')
        tr = next(x for x in root.findall('track') if x.get('task_id') == tid and x.get('label') == 'person')
        hr = next(x for x in root.findall('track') if x.get('task_id') == tid and x.get('label') == 'head')
        rows = []
        for sk, box in zip(tr.findall('skeleton'), hr.findall('box')):
            frame = int(sk.get('frame'))
            assert frame == int(box.get('frame'))
            pts = [{**pt.attrib, 'xy': [float(x) for x in pt.get('points').split(',')]} for pt in sk.findall('points')]
            rows.append(dict(global_image_index=frame, task_local_index=frame-24*ordinal,
                             image=f'images/frame_{frame:06d}.PNG', points=pts, head_box=box.attrib,
                             source_frame=None, source_pts=None, time_base=None))
        assert len(rows)==24 and [x['task_local_index'] for x in rows] == list(range(24))
        tasks.append(dict(sequence_id=f'{ordinal+1:02d}', task_id=tid, source_name=task.findtext('source'),
                          size=[int(task.findtext('original_size/width')),int(task.findtext('original_size/height'))],
                          official_subset=task.findtext('subset'), frames=rows))
    return tasks


def sheet(frames, target, cols=5, cell=(360,300), annotated=False):
    result = Image.new('RGB',(cols*cell[0], ((len(frames)+cols-1)//cols)*cell[1]),'#171b22')
    draw = ImageDraw.Draw(result)
    for i,(label,path,points) in enumerate(frames):
        im=Image.open(path).convert('RGB')
        if annotated:
            d=ImageDraw.Draw(im)
            for pt in points:
                x,y=pt['xy'];c='#ffab40' if pt['occluded']=='1' else '#20ff70'
                d.ellipse((x-5,y-5,x+5,y+5),outline=c,width=3)
        im=ImageOps.contain(im,(cell[0]-8,cell[1]-28))
        x=(i%cols)*cell[0];y=(i//cols)*cell[1]
        result.paste(im,(x+(cell[0]-im.width)//2,y+24))
        draw.text((x+6,y+5),label,fill='white')
    result.save(target,quality=90)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['screen','candidate'])
    ap.add_argument('--sequence',type=int)
    args=ap.parse_args()
    t=time.perf_counter(); cpu=time.process_time()
    items=index(); tasks=annotations()
    if not (OUT/'archive_members.json').exists():
        save('archive_members.json',items)
        save('annotation_mapping.json',tasks)
    selected=tasks if args.mode=='screen' else [tasks[args.sequence-1]]
    ids=[s['frames'][0]['global_image_index'] for s in selected] if args.mode=='screen' else [r['global_image_index'] for r in selected[0]['frames']]
    byname={x['name']:x for x in items}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,byname[f'default/frame_{n:06d}.PNG']):n for n in ids}
        for fut in concurrent.futures.as_completed(futures):
            n=futures[fut]
            try:
                row=fut.result();results.append(row);print(json.dumps(row),flush=True)
            except Exception as exc:
                results.append(dict(index=n,error=str(exc)));print(str(exc),flush=True)
    prefix='screen' if args.mode=='screen' else f'sequence_{args.sequence:02d}'
    images=[]
    for s in selected:
        rows=s['frames'][:1] if args.mode=='screen' else s['frames']
        for r in rows:
            path=OUT/r['image']
            if path.exists():
                with Image.open(path) as im:
                    assert list(im.size)==s['size'],(s['sequence_id'], im.size, s['size'])
                images.append((f"seq{s['sequence_id']} / image{r['global_image_index']:03d} / sample{r['task_local_index']:02d}",path,r['points']))
    if images:
        sheet(images,OUT/f'{prefix}_contact.jpg',cols=5 if args.mode=='screen' else 6)
        if args.mode=='candidate':
            sheet(images,OUT/f'{prefix}_annotations.jpg',cols=6,annotated=True)
    save(f'{prefix}_execution.json',dict(results=results,wall_seconds=time.perf_counter()-t,
         main_process_cpu_seconds=time.process_time()-cpu,successful_images=len(images),gpu_used=False))


if __name__=='__main__':
    main()

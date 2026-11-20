"""Interactive, bounded official BEDLAM2 retrieval; credentials stay in memory.

Reads both credentials with terminal echo disabled. No cookies, login pages,
credentials, request bodies, or authenticated URLs are written to disk.
"""
import argparse
import getpass
import json
import html
import re
from pathlib import Path, PurePosixPath
import tarfile
import time
from urllib.parse import urljoin, urlsplit, parse_qs

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-motion-archive', action='store_true',
                        help='Permit bounded access to the official BEDLAM2 motion TAR; never download the whole archive')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    username = getpass.getpass('BEDLAM2 account (hidden): ')
    password = getpass.getpass('BEDLAM2 password (hidden): ')
    session = requests.Session()
    endpoint = 'https://download.is.tue.mpg.de/download.php'
    records = []
    total = 0
    print('READY: JSON commands on stdin; credentials will not be logged.', flush=True)

    def archive_member(cmd):
        nonlocal total
        sfile=cmd['sfile'];wanted=cmd.get('member','__list_only__');cap=int(cmd.get('max_bytes',256*1024**2))
        motion_archive = args.allow_motion_archive and sfile == 'ground_truth/motions/b2_motions_npz_training.tar'
        if not (sfile.startswith('assets/clothing/') or motion_archive) or '..' in PurePosixPath(sfile).parts:
            raise ValueError('Invalid archive path')
        budget = 100*1024**2 if motion_archive else 1024**3
        if not 0<cap<=(20*1024**2 if motion_archive else 512*1024**2) or total+cap+1024**2>budget:
            raise ValueError('Acquisition budget exceeded')
        wp=PurePosixPath(wanted)
        if wp.is_absolute() or '..' in wp.parts or '\\' in wanted:
            raise ValueError('Unsafe member')
        name=cmd['name'];target=out/name
        if Path(name).name!=name or target.exists():raise ValueError('Invalid destination')
        start=time.monotonic()
        response=session.post(endpoint,params={'domain':'bedlam2','sfile':sfile},
            data={'username':username,'password':password},stream=True,timeout=(15,45),allow_redirects=False)
        if response.status_code!=302:
            response.close();raise ValueError('Expected authenticated file redirect')
        signed_url=urljoin(response.url,response.headers.get('Location',''));response.close()
        parts=urlsplit(signed_url)
        if parts.scheme!='https' or parts.hostname!='download.is.tue.mpg.de' or parts.path!='/download2.php':
            raise ValueError('Unexpected file endpoint')
        ranges=[];scanned=[];window_offset=-1;window=b''
        def read_range(offset,size):
            nonlocal total,window_offset,window
            if window_offset<=offset and offset+size<=window_offset+len(window):
                return window[offset-window_offset:offset-window_offset+size]
            request_size=max(size,4096)
            with session.get(signed_url,headers={'Range':f'bytes={offset}-{offset+request_size-1}'},
                             stream=True,timeout=(15,45),allow_redirects=False) as r:
                if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-{offset+request_size-1}/'):
                    raise ValueError('Exact Range not honored; body not downloaded')
                data=r.raw.read(request_size+1)
                if len(data)!=request_size:raise ValueError('Truncated or oversized range')
                total+=len(data);ranges.append(dict(offset=offset,bytes=request_size))
                window_offset,window=offset,data
                return data[:size]
        offset=int(cmd.get('start_offset',0));pax={};longname=None
        if offset<0 or offset%512:raise ValueError('Invalid initial TAR offset')
        for i in range(int(cmd.get('max_entries',512))):
            block=read_range(offset,512)
            if not any(block):break
            h=tarfile.TarInfo.frombuf(block,'utf8','strict')
            size=h.size
            if h.type in (tarfile.XHDTYPE,tarfile.XGLTYPE):
                if size>65536:raise ValueError('Unexpected large PAX header')
                data=read_range(offset+512,size);pos=0;parsed={}
                while pos<len(data):
                    space=data.index(b' ',pos);length=int(data[pos:space]);entry=data[space+1:pos+length-1]
                    k,v=entry.decode('utf8').split('=',1);parsed[k]=v;pos+=length
                pax.update(parsed)
            elif h.type==tarfile.GNUTYPE_LONGNAME:
                if size>4096:raise ValueError('Unexpected long name')
                longname=read_range(offset+512,size).rstrip(b'\0').decode('utf8')
            else:
                member=pax.get('path',longname or h.name)
                size=int(pax.get('size',size));pax={};longname=None
                scanned.append(dict(name=member,size=size,offset=offset))
                if len(scanned)%100==0:print(json.dumps(dict(archive_entries_checked=len(scanned),header_bytes=sum(x['bytes'] for x in ranges),last_member=member,offset=offset)),flush=True)
                if cmd.get('op') != 'list_archive' and PurePosixPath(member)==wp:
                    if not h.isfile() or not 0<size<=cap:raise ValueError('Invalid selected member size/type')
                    chunk_bytes=int(cmd.get('chunk_bytes',0))
                    if chunk_bytes:
                        if not 0<chunk_bytes<=8*1024**2:raise ValueError('Invalid chunk size')
                        partial=out/(name+'.part')
                        checkpoint=out/(name+'.partial_progress.json')
                        if partial.exists():
                            if not cmd.get('resume') or not checkpoint.exists():raise FileExistsError(partial)
                            previous=json.loads(checkpoint.read_text())
                            if previous.get('sfile')!=sfile or previous.get('member')!=member or previous.get('member_bytes')!=size:
                                raise ValueError('Partial member mismatch')
                            written=partial.stat().st_size
                            if previous.get('written_bytes')!=written or written>=size:raise ValueError('Partial size mismatch')
                        else:
                            if cmd.get('resume'):raise ValueError('Missing partial for resume')
                            partial.open('xb').close();written=0
                        with partial.open('ab') as f:
                            while written<size:
                                n=min(chunk_bytes,size-written)
                                data=read_range(offset+512+written,n)
                                f.write(data);f.flush()
                                written+=n
                                checkpoint.write_text(json.dumps(dict(sfile=sfile,member=member,member_bytes=size,
                                    written_bytes=written,chunk_bytes=chunk_bytes))+'\n')
                                print(json.dumps(dict(member_progress_bytes=written,member_bytes=size)),flush=True)
                        partial.rename(target)
                    else:
                        payload=read_range(offset+512,size)
                        with target.open('xb') as f:f.write(payload)
                    record=dict(sfile=sfile,member=member,file=str(target),member_bytes=size,
                        transferred_bytes=sum(x['bytes'] for x in ranges),ranges=ranges,entries=scanned,
                        seconds=time.monotonic()-start,downloaded_unreviewed=True)
                    records.append(record)
                    (out/(name+'.acquisition.json')).write_text(json.dumps(record,indent=2)+'\n')
                    print(json.dumps({k:record[k] for k in ['sfile','member','file','member_bytes','transferred_bytes','seconds']}),flush=True)
                    return
            offset+=512+((size+511)//512)*512
        if cmd.get('op') == 'list_archive':
            print(json.dumps(dict(entries=scanned,next_offset=offset,transferred_bytes=sum(x['bytes'] for x in ranges))),flush=True)
            return
        print(json.dumps(dict(member_not_found_within_entries=len(scanned),next_offset=offset,transferred_bytes=sum(x['bytes'] for x in ranges))),flush=True)
        raise ValueError('Member not found within bounded archive headers')

    def catalog():
        site='https://bedlam2.is.tuebingen.mpg.de'
        with session.post(site+'/login.php',data={'username':username,'password':password,'commit':'Log in'},
                          timeout=(15,45),allow_redirects=False) as r:
            if r.status_code not in (200,302,303):raise ValueError('Website login not accepted')
        with session.get(site+'/download.php',timeout=(15,45),allow_redirects=False) as r:
            if r.status_code!=200:raise ValueError('Authenticated catalog unavailable')
            page=r.text
        links=[]
        for match in re.finditer(r'href=[\"\x27]([^\"\x27]+)[\"\x27]',page):
            url=html.unescape(match.group(1));parts=urlsplit(url)
            if parts.hostname!='download.is.tue.mpg.de':continue
            query=parse_qs(parts.query);sfile=query.get('sfile',[''])[0]
            if query.get('domain')!=['bedlam2'] or not sfile.startswith('assets/clothing/'):continue
            context=html.unescape(re.sub('<[^>]*>',' ',page[match.end():match.end()+250]))
            context=' '.join(context.split()).replace(username,'[redacted]').replace(password,'[redacted]')
            links.append(dict(sfile=sfile,description_excerpt=context[:160]))
        if not links:raise ValueError('No clothing catalog links found')
        (out/'official_clothing_links.json').write_text(json.dumps(links,indent=2)+'\n')
        print(json.dumps(links),flush=True)

    def retrieve(cmd):
        nonlocal total
        sfile = cmd['sfile']
        path = PurePosixPath(sfile)
        motion_probe = args.allow_motion_archive and cmd.get('op') == 'probe' and sfile == 'ground_truth/motions/b2_motions_npz_training.tar'
        if not (sfile.startswith('assets/clothing/') or motion_probe) or '..' in path.parts or path.is_absolute():
            raise ValueError('Only declared clothing paths are allowed')
        cap = int(cmd.get('max_bytes', 8 * 1024**2))
        if not 0 < cap <= 512 * 1024**2 or total + cap > 1024**3:
            raise ValueError('Acquisition byte budget exceeded')
        headers = {'Referer': 'https://bedlam2.is.tuebingen.mpg.de/download.php'}
        if 'range' in cmd:
            offset, length = map(int, cmd['range'])
            if offset < 0 or length <= 0 or length > cap:
                raise ValueError('Invalid range')
            headers['Range'] = f'bytes={offset}-{offset+length-1}'
        started = time.monotonic()
        response = session.post(endpoint, params={'domain':'bedlam2','sfile':sfile},
                          data={'username':username,'password':password},
                          headers=headers, stream=True, timeout=(15,45),
                          allow_redirects=False)
        if response.status_code in (301,302,303,307,308):
            location=urljoin(response.url,response.headers.get('Location',''))
            parts=urlsplit(location)
            print(json.dumps(dict(redirect_status=response.status_code,
                                  redirect_host=parts.hostname,redirect_path=parts.path,
                                  redirect_query_redacted=True)),flush=True)
            response.close()
            allowed={'download.is.tue.mpg.de','bedlam2.is.tuebingen.mpg.de','bedlam2.is.tue.mpg.de'}
            if parts.scheme!='https' or parts.hostname not in allowed or 'login' in parts.path.lower():
                print(json.dumps(dict(result='redirect_not_followed_no_asset_download')),flush=True)
                return
            response=session.get(location,headers=headers,stream=True,timeout=(15,45),allow_redirects=False)
        with response:
            status = response.status_code
            content_type = response.headers.get('Content-Type','')
            length = response.headers.get('Content-Length')
            summary = dict(sfile=sfile,status=status,content_type=content_type,
                           content_length=int(length) if length and length.isdigit() else None,
                           content_range=response.headers.get('Content-Range'),
                           content_disposition=response.headers.get('Content-Disposition'))
            if status not in (200,206) or 'text/html' in content_type.lower():
                # Never persist the response body: it could contain a login form.
                print(json.dumps(dict(summary,result='not_asset_response_no_body_saved')),flush=True)
                return
            if cmd.get('op') == 'probe':
                print(json.dumps(dict(summary,result='headers_only_body_not_downloaded')),flush=True)
                records.append(summary)
                return
            if 'range' in cmd and status != 206:
                raise ValueError('Server ignored Range; body not downloaded')
            if summary['content_length'] is not None and summary['content_length'] > cap:
                raise ValueError('Server length exceeds frozen cap')
            name = cmd['name']
            if Path(name).name != name or name in ('','.', '..'):
                raise ValueError('Output must be a simple basename')
            target = out/name
            if target.exists():
                raise FileExistsError('Output already exists')
            chunks=[];count=0
            for chunk in response.iter_content(1024*1024):
                count += len(chunk)
                if count > cap:
                    raise ValueError('Response exceeded byte cap')
                chunks.append(chunk)
            payload=b''.join(chunks)
            if cmd.get('json',False):
                data=json.loads(payload)
                if not isinstance(data,(dict,list)):
                    raise ValueError('Unexpected JSON structure')
                # Only asset JSON is persisted; credentials are never included.
            with target.open('xb') as handle:
                handle.write(payload)
            total+=count
            record=dict(summary,file=str(target),bytes=count,seconds=time.monotonic()-started,
                        downloaded_unreviewed=True)
            records.append(record)
            print(json.dumps(record),flush=True)
        (out/'transfer_log.json').write_text(json.dumps(records,indent=2)+'\n')

    while True:
        try:
            line=input()
        except EOFError:
            break
        if line.strip()=='quit':
            break
        try:
            cmd=json.loads(line)
            if cmd.get('op') in ('member','list_archive'):archive_member(cmd)
            elif cmd.get('op')=='catalog':catalog()
            else:retrieve(cmd)
        except (requests.RequestException,ValueError,KeyError,FileExistsError) as error:
            # Exception messages may contain URL/server data; print only category.
            print(json.dumps({'error_type':type(error).__name__,'response_not_accepted':True}),flush=True)
    session.close()


if __name__=='__main__':
    main()

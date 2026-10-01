"""Contact sheets at full 256-square scale; does not approve any candidate."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--layouts',nargs='+',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    out=a.root/'review'/a.name;out.mkdir(parents=True,exist_ok=False);tiles=[];erp=[];index=[]
    for name in a.layouts:
        directory=a.root/'layouts'/name
        if json.loads((directory/'validation.json').read_text())['status']!='passed':raise ValueError(name)
        people={p['id']:p for p in json.loads((directory/'people.json').read_text())['people']}
        with np.load(directory/'diagnostic_loader.npz') as z:
            for pid,cam in zip(z['person_id'],z['camera_id']):
                pid=int(pid);cam=str(cam);person=people[pid];source=directory/'actual_inputs'/f'{cam}_person_{pid}_square.png'
                tile=Image.new('RGB',(256,288),'white');tile.paste(Image.open(source),(0,0));draw=ImageDraw.Draw(tile)
                draw.text((3,258),f'{name} {cam}/{pid} {person["character_id"].split("_")[1]}',fill='black')
                draw.text((3,272),f'{person["action"]} {person["category"]}',fill='black')
                # Lines mark network columns; do not alter crop pixels themselves.
                for x in (32,224):draw.line((x,253,x,257),fill='red',width=2)
                tiles.append(tile);index.append(dict(layout=name,camera=cam,person_id=pid,square=str(source)))
        for cam in ['cam2','cam4']:
            im=Image.open(directory/cam/'normal_erp.png').convert('RGB');im.thumbnail((768,384))
            tile=Image.new('RGB',(768,406),'white');tile.paste(im,(0,0));ImageDraw.Draw(tile).text((3,388),f'{name} {cam}',fill='black');erp.append(tile)
    for page,start in enumerate(range(0,len(tiles),24)):
        selected=tiles[start:start+24];sheet=Image.new('RGB',(1536,288*((len(selected)+5)//6)),'white')
        for i,tile in enumerate(selected):sheet.paste(tile,(i%6*256,i//6*288))
        sheet.save(out/f'inputs_{page:02d}.jpg',quality=95)
    for page,start in enumerate(range(0,len(erp),8)):
        selected=erp[start:start+8];sheet=Image.new('RGB',(1536,406*((len(selected)+1)//2)),'white')
        for i,tile in enumerate(selected):sheet.paste(tile,(i%2*768,i//2*406))
        sheet.save(out/f'erp_{page:02d}.jpg',quality=92)
    (out/'index.json').write_text(json.dumps(dict(layouts=a.layouts,rows=index,status='awaiting_visual_review'),indent=2)+'\n')
    print(out)

if __name__=='__main__':main()

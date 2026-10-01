"""Contact sheets link actual render files; no generated or retouched imagery."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw

def sheet(rows,out,width=300,height=320):
    canvas=Image.new('RGB',(width*5,height*((len(rows)+4)//5)),(32,32,32))
    for i,(label,path) in enumerate(rows):
        image=Image.open(path).convert('RGB');image.thumbnail((width,height-24))
        x=i%5*width;y=i//5*height
        canvas.paste(image,(x+(width-image.width)//2,y+24))
        ImageDraw.Draw(canvas).text((x+5,y+5),label,fill='white')
    canvas.save(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path);p.add_argument('--outfits',action='store_true');a=p.parse_args();r=a.assets.resolve()
    chars=sorted(json.loads((r/'report.json').read_text())['characters'],key=lambda x:x['character_id']);sources=[]
    for kind in ['head_front','front','head_clay']:
        rows=[]
        for char in chars:
            cid=char['character_id'];short=cid.split('_')[1]
            if short=='F04' and kind!='head_clay':folder=r/'uniform_F04_v3_review'/cid/'reference'
            elif short=='F05' and kind!='head_clay':folder=r/'uniform_revision_review'/cid/'reference'
            else:folder=next((r/'uniform_review').glob(f'cycling_{short}_*/reference'))
            path=folder/(kind+'.png');rows.append((cid,path));sources.append(dict(asset=cid,view=kind,file=str(path),geometry_reused=short in ['F04','F05'] and kind=='head_clay'))
        sheet(rows,r/f'identity_{kind}.jpg')
    if a.outfits:
        for view in ['head_front','front','side','back']:
            rows=[]
            for char in chars:
                cid=char['character_id'];pose='reference' if view=='head_front' else 'coco_292226_person_1330527'
                folder=r/('outfit_female' if cid.split('_')[1][0]=='F' else 'outfit_male')/cid/pose
                path=folder/(view+'.png');rows.append((cid,path));sources.append(dict(asset=cid,view=view,file=str(path)))
            sheet(rows,r/f'outfit_{view}.jpg')
    (r/'review_sheet_sources.json').write_text(json.dumps(sources,indent=2)+'\n')

if __name__=='__main__':main()

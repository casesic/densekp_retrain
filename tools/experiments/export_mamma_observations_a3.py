"""Export an existing MAMMA ROI run; does not start rendering."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.data.mamma_export import export_existing


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New manifest JSON path')
    parser.add_argument('--qualification-registry',type=Path,required=True,
                        help='Reviewed A1 reader config or handoff with qualified_assets map')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=export_existing(args.run,args.qualification_registry)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(manifest=str(args.output.resolve()),slots=len(result['observations']),
        observed=sum(r['status']=='observed' for r in result['observations']),scope=result['frame_scope'])))


if __name__=='__main__':main()

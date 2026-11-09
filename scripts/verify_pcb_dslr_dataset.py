#!/usr/bin/env python3
"""Audit PCB-DSLR source hashes, YOLO labels, and board-level split before training."""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST=ROOT/'data/component_training/manifests/pcb-dslr-part1-ic-manifest.json'
DEFAULT_DATA=ROOT/'data/component_training/processed/pcb-dslr-part1-ic-v3/data.yaml'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_data_paths(data_path):
    """Read the small path section of a YOLO data YAML without PyYAML."""
    values={}
    for line in data_path.read_text().splitlines():
        if line[:1].isspace() or ':' not in line:continue
        key,value=line.split(':',1);key=key.strip();value=value.strip()
        if key in {'path','train','val','test'}:
            if value.startswith('"') and value.endswith('"'):
                value=json.loads(value)
            elif value.startswith("'") and value.endswith("'"):
                value=value[1:-1]
            values[key]=value
    if 'path' not in values:raise ValueError(f'missing path in YOLO data file: {data_path}')
    return values
def check(manifest_path=DEFAULT_MANIFEST,data_path=DEFAULT_DATA,root=ROOT,strict_counts=True):
    manifest=json.loads(manifest_path.read_text());data=read_data_paths(data_path)
    base=Path(data['path']);base=base if base.is_absolute() else (data_path.parent/base).resolve()
    errors=[];by_split=Counter();boards=defaultdict(set);boxes=Counter();class_counts=Counter()
    if manifest.get('dataset_license')!='FREE_FOR_NON_COMMERCIAL_RESEARCH_USE_ONLY':errors.append('license is not explicitly non-commercial research-only')
    if manifest.get('training_ready') is not True:errors.append('manifest is not training ready')
    for row in manifest['images']:
        split=row['export_split'];by_split[split]+=1;boards[row['board_group_id']].add(split);boxes[split]+=row['annotation_count']
        if row.get('training_include') is not True:errors.append(f"manifest excludes {row['image_id']}")
        for rel,key in ((row['image_path'],'image_sha256'),(row['annotation_path'],'annotation_sha256'),(row['board_mask_path'],'mask_sha256'),(row['exported_label_path'],'label_sha256')):
            source=root/rel
            if not source.is_file():errors.append(f"missing {rel}");continue
            if sha(source)!=row[key]:errors.append(f"hash mismatch {rel}")
        image=root/row['image_path']
        if image.is_file():
            with Image.open(image) as im:
                if list(im.size)!=row['image_size_px']:errors.append(f"image dimensions differ {row['image_id']}")
        label=root/row['exported_label_path']
        if label.is_file():
            lines=[x.split() for x in label.read_text().splitlines() if x.strip()]
            if len(lines)!=row['annotation_count']:errors.append(f"box count mismatch {row['image_id']}")
            for fields in lines:
                if len(fields)!=5 or fields[0]!='0':errors.append(f"invalid one-class YOLO row {row['image_id']}");continue
                try:cls,cx,cy,w,h=int(fields[0]),*map(float,fields[1:])
                except ValueError:errors.append(f"invalid numeric label {row['image_id']}");continue
                if not all(0<=v<=1 for v in (cx,cy,w,h)) or w<=0 or h<=0 or cx-w/2<0 or cy-h/2<0 or cx+w/2>1 or cy+h/2>1:errors.append(f"out-of-bounds box {row['image_id']}")
                class_counts[manifest['class_names'][cls]]+=1
    leaked={b:sorted(s) for b,s in boards.items() if len(s)>1}
    if leaked:errors.append(f'board split leakage: {leaked}')
    expected={'train':('train',59,14,1825),'validation':('val',14,3,153),'test':('test',14,3,154)}
    for split,(yaml_key,images,board_n,box_n) in expected.items():
        if strict_counts:
            if by_split[split]!=images:errors.append(f'{split} image count {by_split[split]} != {images}')
            if len([b for b,s in boards.items() if split in s])!=board_n:errors.append(f'{split} board count mismatch')
            if boxes[split]!=box_n:errors.append(f'{split} box count mismatch')
        folder=base/data[yaml_key]
        if not folder.is_dir():errors.append(f'missing YAML {yaml_key} directory {folder}')
    return {'status':'PASS' if not errors else 'BLOCKED','errors':errors,'dataset_id':manifest['dataset_id'],'dataset_version':manifest['source_version'],'license_scope':manifest['license_scope'],'manifest_training_ready':manifest['training_ready'],'manifest_status':manifest['status'],'images':dict(by_split),'boards':{'train':len({b for b,s in boards.items() if 'train' in s}),'validation':len({b for b,s in boards.items() if 'validation' in s}),'test':len({b for b,s in boards.items() if 'test' in s}),'total':len(boards)},'annotations':dict(boxes),'class_counts':dict(class_counts),'board_leakage':leaked,'data_yaml':str(data_path.resolve()),'data_yaml_sha256':sha(data_path),'manifest_sha256':sha(manifest_path)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST);p.add_argument('--data',type=Path,default=DEFAULT_DATA);p.add_argument('--output',type=Path)
    a=p.parse_args();report=check(a.manifest,a.data)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())

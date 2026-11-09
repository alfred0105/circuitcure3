#!/usr/bin/env python3
"""Build a leak-conservative exploratory YOLO split from C.1 source annotations."""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/component_training/manifests/pcb-component-detection-curation-manifest.json'
IDENTITY=ROOT/'data/component_training/manifests/pcb-component-detection-c1-identity-annotation-report.json'
OUTPUT=ROOT/'data/component_training/processed/kaggle-source-only-yolov8s-dev-v1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def check_bbox(box,size):
 if len(box)!=4:return False
 x1,y1,x2,y2=map(float,box);w,h=size
 return 0<=x1<x2<=w and 0<=y1<y2<=h
def build(manifest_path=MANIFEST,identity_path=IDENTITY,output=OUTPUT):
 manifest=json.loads(manifest_path.read_text());identity=json.loads(identity_path.read_text());rows=manifest['images']
 if output.exists() and any(output.iterdir()):raise FileExistsError(f'output must be empty: {output}')
 classes=list(manifest['source_classes']);index={name:i for i,name in enumerate(classes)}
 parent={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[b]=a
 by_stem=defaultdict(list)
 for r in rows:by_stem[r['source_stem_group']].append(r)
 for stem,members in by_stem.items():
  find(stem)
  for r in members:union(stem,'IMG:'+r['image_id'])
 by_hash={}
 for r in rows:
  path=ROOT/r['image_path']
  if not path.is_file():raise FileNotFoundError(path)
  actual=sha(path)
  if actual!=r['image_sha256']:raise ValueError(f"image hash mismatch: {r['image_id']}")
  if actual in by_hash:union('IMG:'+r['image_id'],'IMG:'+by_hash[actual])
  else:by_hash[actual]=r['image_id']
  if r.get('augmentation_family_id'):union(r['source_stem_group'],'AUG:'+r['augmentation_family_id'])
 for pair in identity['orb_ransac_review_candidates']:
  union(pair['first_stem_group'],pair['second_stem_group'])
 components=defaultdict(list)
 for r in rows:components[find('IMG:'+r['image_id'])].append(r)
 labelled=[r for r in rows if r['annotations']]
 class_components=defaultdict(set)
 for root,members in components.items():
  for r in members:
   for a in r['annotations']:class_components[a['class']].add(root)
 unique_support={name:next(iter(groups)) for name,groups in class_components.items() if len(groups)==1}
 forced_train=set(unique_support.values())
 # The largest ORB-connected sequence is a source video cluster; keep it on
 # the training side so one near-duplicate sequence cannot dominate dev data.
 largest_component=max(components,key=lambda key:sum(bool(r['annotations']) for r in components[key]))
 forced_train.add(largest_component)
 target=max(1,round(len(labelled)*.2));selected=[];selected_images=0;val_classes=set()
 candidates=[]
 for root,members in components.items():
  l=[r for r in members if r['annotations']]
  if not l or root in forced_train:continue
  class_set={a['class'] for r in l for a in r['annotations']}
  score=hashlib.sha256(('phase21-3-2-d:'+root).encode()).hexdigest()
  candidates.append((root,l,class_set,score))
 candidates.sort(key=lambda x:(-len(x[2]-val_classes),x[3]))
 for root,l,cset,_ in candidates:
  if selected_images>=target:break
  if any(len(class_components[c]-{root}-set(selected))==0 for c in cset):continue
  selected.append(root);selected_images+=len(l);val_classes.update(cset)
 val_roots=set(selected)
 if not val_roots:raise ValueError('could not construct an exploratory validation group split')
 split_counts=Counter();box_counts=Counter();split_boards=defaultdict(set);errors=[]
 output.mkdir(parents=True,exist_ok=True)
 for split in ('train','validation'):
  (output/'images'/split).mkdir(parents=True,exist_ok=True);(output/'labels'/split).mkdir(parents=True,exist_ok=True)
 for r in labelled:
  root=find('IMG:'+r['image_id']);split='validation' if root in val_roots else 'train'
  img_src=ROOT/r['image_path'];lab_src=ROOT/r['label_path'];
  if not lab_src.is_file():raise FileNotFoundError(lab_src)
  if sha(lab_src)!=r['label_sha256']:raise ValueError(f"label hash mismatch: {r['image_id']}")
  w,h=r['image_size_px'];lines=[]
  for ann in r['annotations']:
   cls=ann['class'];box=ann['bbox_xyxy']
   if cls not in index or not check_bbox(box,(w,h)):errors.append(f"invalid label {r['image_id']} {cls} {box}");continue
   x1,y1,x2,y2=map(float,box);cx=(x1+x2)/(2*w);cy=(y1+y2)/(2*h);bw=(x2-x1)/w;bh=(y2-y1)/h
   lines.append(f'{index[cls]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}')
   box_counts[(split,cls)]+=1
  stem=Path(r['image_id']).stem
  out_img=output/'images'/split/(stem+Path(r['image_path']).suffix.lower());out_lab=output/'labels'/split/(stem+'.txt')
  out_img.symlink_to(img_src.resolve());out_lab.write_text('\n'.join(lines)+'\n')
  split_counts[split]+=1;split_boards[split].add(root)
 if errors:raise ValueError('\n'.join(errors[:30]))
 if set(split_boards['train']) & set(split_boards['validation']):raise AssertionError('group leakage')
 data={'path':str(output.resolve()),'train':'images/train','val':'images/validation','names':{i:name for i,name in enumerate(classes)}}
 yaml_lines=[f'path: {json.dumps(data["path"],ensure_ascii=False)}','train: images/train','val: images/validation','names:']
 yaml_lines.extend(f'  {i}: {json.dumps(name,ensure_ascii=False)}' for i,name in enumerate(classes))
 (output/'data.yaml').write_text('\n'.join(yaml_lines)+'\n')
 n_empty=len(rows)-len(labelled)
 group_info={'train_component_groups':len(split_boards['train']),'validation_component_groups':len(split_boards['validation']),
   'train_filename_groups':len({r['source_stem_group'] for r in labelled if find('IMG:'+r['image_id']) not in val_roots}),
   'validation_filename_groups':len({r['source_stem_group'] for r in labelled if find('IMG:'+r['image_id']) in val_roots})}
 report={'schema_version':'circuitcare.kaggle-source-only-experiment.v1','status':'EXPLORATORY_SOURCE_LABELS',
   'source_manifest_sha256':sha(manifest_path),'identity_report_sha256':sha(identity_path),
   'source_classes':classes,'image_count_before':len(rows),'empty_annotation_excluded':n_empty,'image_count_used':len(labelled),
   'annotation_count_used':sum(box_counts.values()),'split_images':dict(split_counts),'split_groups':group_info,
   'split_policy':'All filename stems, declared augmentation families, exact image SHA duplicates, and all C.1 ORB/RANSAC candidate pairs are kept in one connected split group.',
   'source_split_used_for_partition':False,'physical_board_ids':'UNVERIFIED','sealed_test_split':False,
   'source_only_labels':True,'canonical_class_merges':[],'class_box_counts':{split:{c:box_counts[(split,c)] for c in classes} for split in ('train','validation')},
   'classes_missing_from_validation':[c for c in classes if box_counts[('validation',c)]==0],
   'source_split_original_partition_ignored':True,
   'original_source_split_conflicting_filename_groups':sum(len({r['source_split'] for r in members})>1 for members in by_stem.values()),
   'limits':['Source annotations are not fully reviewed or guaranteed complete.','No physical board identity or independent sealed test is available.','The connected duplicate/ORB groups are conservative visual groups, not verified board IDs.'],
   'data_yaml':str((output/'data.yaml').resolve())}
 (output/'experiment_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=MANIFEST);p.add_argument('--identity-report',type=Path,default=IDENTITY);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();r=build(a.manifest,a.identity_report,a.output);print(json.dumps(r,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())

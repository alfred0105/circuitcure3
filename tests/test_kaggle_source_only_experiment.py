import hashlib, importlib.util, json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('kaggle_exp',ROOT/'scripts/prepare_kaggle_source_only_experiment.py')
PREP=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(PREP)

def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_experiment_keeps_duplicate_groups_together_and_excludes_empty(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();rows=[]
    definitions=[('a1','g-a',1),('a2','g-a',1),('b1','g-b',1),('d1','g-d',1),('empty','g-c',0)]
    for idx,(name,group,has_box) in enumerate(definitions):
        image=raw/(name+'.jpg');Image.new('RGB',(40,30),(30+idx,40,90)).save(image)
        label=raw/(name+'.json');label.write_text('{}')
        anns=[{'class':'Resistor','bbox_xyxy':[2,3,15,18]}] if has_box else []
        rows.append({'image_id':name+'.jpg','image_path':str(image),'label_path':str(label),'image_sha256':h(image),'label_sha256':h(label),'source_stem_group':group,'augmentation_family_id':group,'source_split':'train' if name in ('a1','b1') else 'validation','image_size_px':[40,30],'annotation_count':len(anns),'annotations':anns})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'source_classes':['Resistor'],'split_policy':'source split ignored','images':rows}))
    identity=tmp_path/'identity.json';identity.write_text(json.dumps({'orb_ransac_review_candidates':[{'first_stem_group':'g-a','second_stem_group':'g-b'}]}))
    out=tmp_path/'out';result=PREP.build(manifest,identity,out)
    assert result['image_count_used']==4 and result['empty_annotation_excluded']==1
    assert result['sealed_test_split'] is False
    splits={}
    for split in ('train','validation'):
        for img in (out/'images'/split).iterdir():splits[img.stem]=split
    assert splits['a1']==splits['a2']==splits['b1'] and splits['d1']!=splits['a1']
    assert not any('empty' in p.name for folder in (out/'images/train',out/'images/validation') for p in folder.iterdir())
    assert len(result['source_classes'])==1

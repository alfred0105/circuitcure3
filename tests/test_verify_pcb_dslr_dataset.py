import hashlib, importlib.util, json
import pytest
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('verify_pcb_dslr',ROOT/'scripts/verify_pcb_dslr_dataset.py')
VERIFY=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(VERIFY)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def test_data_yaml_path_reader_handles_quoted_and_unquoted_values(tmp_path):
    data=tmp_path/'data.yaml';data.write_text(f'path: "{tmp_path} with spaces"\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n  0: integrated_circuit\n')
    assert VERIFY.read_data_paths(data)=={'path':f'{tmp_path} with spaces','train':'images/train','val':'images/validation','test':'images/test'}

def test_dataset_audit_detects_bad_hash_and_board_leakage(tmp_path):
    rows=[]
    for i,split in enumerate(('train','validation')):
        board='same-board' if i==0 else 'other-board'
        source=tmp_path/'raw'/f'board{i}.jpg';source.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(20,16),(10,20,i+30)).save(source)
        ann=tmp_path/'raw'/f'board{i}.txt';ann.write_text('annotation source')
        mask=tmp_path/'raw'/f'board{i}-mask.png';Image.new('L',(20,16),255).save(mask)
        label=tmp_path/'processed'/'labels'/split/f'board{i}.txt';label.parent.mkdir(parents=True,exist_ok=True);label.write_text('0 0.50000000 0.50000000 0.50000000 0.50000000\n')
        rows.append({'image_id':f'board{i}','image_path':str(source.relative_to(tmp_path)),'annotation_path':str(ann.relative_to(tmp_path)),'board_mask_path':str(mask.relative_to(tmp_path)),'image_sha256':digest(source),'annotation_sha256':digest(ann),'mask_sha256':digest(mask),'exported_label_path':str(label.relative_to(tmp_path)),'label_sha256':digest(label),'image_size_px':[20,16],'export_split':split,'board_group_id':board,'training_include':True,'annotation_count':1,'annotations':[{'class':'integrated_circuit','bbox_xyxy':[5,4,15,12]}]})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'dataset_id':'fixture','source_version':'fixture','dataset_license':'FREE_FOR_NON_COMMERCIAL_RESEARCH_USE_ONLY','license_scope':'research','training_ready':True,'status':'TRAINING_READY','class_names':['integrated_circuit'],'images':rows}))
    for s in ('train','validation','test'):(tmp_path/'images'/s).mkdir(parents=True,exist_ok=True)
    data=tmp_path/'data.yaml';data.write_text(f'path: {tmp_path}\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n  0: integrated_circuit\n')
    result=VERIFY.check(manifest,data,root=tmp_path,strict_counts=False)
    assert result['status']=='PASS',result['errors']
    rows[1]['board_group_id']='same-board';manifest.write_text(json.dumps({'dataset_id':'fixture','source_version':'fixture','dataset_license':'FREE_FOR_NON_COMMERCIAL_RESEARCH_USE_ONLY','license_scope':'research','training_ready':True,'status':'TRAINING_READY','class_names':['integrated_circuit'],'images':rows}))
    result=VERIFY.check(manifest,data,root=tmp_path,strict_counts=False)
    assert result['status']=='BLOCKED' and any('leakage' in error for error in result['errors'])

def test_checked_in_pcb_dslr_manifest_and_local_dataset_are_consistent():
    local_manifest=ROOT/'data/component_training/manifests/pcb-dslr-part1-ic-manifest.json'
    local_image=ROOT/'data/component_training/raw/external/pcb-dslr/part1-extracted/pcb1/rec1.jpg'
    if not local_manifest.is_file() or not local_image.is_file():pytest.skip('local non-commercial dataset files are not part of CI')
    report=VERIFY.check()
    assert report['status']=='PASS',report['errors']
    assert report['images']=={'train':59,'test':14,'validation':14}
    assert report['boards']=={'train':14,'validation':3,'test':3,'total':20}
    assert sum(report['annotations'].values())==2132
    assert report['board_leakage']=={}

#!/usr/bin/env python3
"""Run a development-only Ultralytics evaluation for source-label Kaggle experiments."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--experiment-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('mps','cpu'),default='mps');p.add_argument('--imgsz',type=int,default=640);a=p.parse_args()
 import torch
 from ultralytics import YOLO
 if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
 exp=json.loads(a.experiment_manifest.read_text());model=YOLO(str(a.weights))
 val_dir=a.output.resolve()
 val=model.val(data=str(a.data.resolve()),split='val',imgsz=a.imgsz,device=a.device,plots=True,verbose=True,project=str(val_dir.parent),name=val_dir.name,exist_ok=False)
 names=val.names;box=val.box;present=set(int(x) for x in getattr(val,'ap_class_index',[]).tolist()) if hasattr(getattr(val,'ap_class_index',None),'tolist') else set()
 per_class={}
 for i,name in (names.items() if isinstance(names,dict) else enumerate(names)):
  available=i in present
  row={'validation_boxes':exp['class_box_counts']['validation'].get(name,0),'precision':None,'recall':None,'AP50':None,'AP50_95':None,'status':'NO_VALIDATION_SUPPORT' if not available else 'MEASURED_SOURCE_LABELS'}
  if available:
   try:
    pvals=box.p;rvals=box.r;ap50=box.ap50;ap=box.ap
    ix=list(val.ap_class_index).index(i)
    row.update({'precision':float(pvals[ix]),'recall':float(rvals[ix]),'AP50':float(ap50[ix]),'AP50_95':float(ap[ix])})
   except (AttributeError,IndexError,ValueError,TypeError):pass
  per_class[name]=row
 report={'schema_version':'circuitcare.kaggle-source-only-evaluation.v1','status':'DEVELOPMENT_ONLY_SOURCE_LABEL_METRICS','dataset_status':exp['status'],'images':exp['split_images'],'groups':exp['split_groups'],'physical_boards':'UNVERIFIED','sealed_test':'NOT_AVAILABLE','empty_annotation_excluded':exp['empty_annotation_excluded'],'source_classes':exp['source_classes'],'model':str(a.weights.resolve()),'imgsz':a.imgsz,'device':a.device,'confidence_threshold':'Ultralytics validation default; see evaluator version and run args','overall':{'precision':float(box.mp),'recall':float(box.mr),'mAP50':float(box.map50),'mAP50_95':float(box.map)},'per_class':per_class,'caveats':exp['limits']+['These metrics are agreement with source labels, not adjudicated truth.','Connected visual groups do not establish physical PCB identity.','Do not compare directly with the single-class PCB-DSLR metrics.'],'ultralytics_artifacts_dir':str(val_dir)}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps(report,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Compare post-NMS boxes and scores from a PyTorch checkpoint and its ONNX export."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image

def iou(a,b):
 x1,y1=np.maximum(a[:2],b[:2]);x2,y2=np.minimum(a[2:],b[2:]);inter=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
 return float(inter/(aa+bb-inter)) if aa+bb>inter else 0.0

def compare_rows(pt,onnx,minimum_iou):
 unused=set(range(len(onnx)));matches=[]
 for pred in sorted(pt,key=lambda r:-r['confidence']):
  candidates=[(iou(pred['bbox_xyxy'],onnx[j]['bbox_xyxy']),j) for j in unused if pred['class_id']==onnx[j]['class_id']]
  if not candidates:continue
  overlap,index=max(candidates)
  if overlap>=minimum_iou:
   unused.remove(index);other=onnx[index]
   matches.append({'class_id':pred['class_id'],'iou':overlap,'confidence_abs_delta':abs(pred['confidence']-other['confidence']),'bbox_max_abs_delta_px':max(abs(a-b) for a,b in zip(pred['bbox_xyxy'],other['bbox_xyxy']))})
 return {'pytorch_detection_count':len(pt),'onnx_detection_count':len(onnx),'matched_count':len(matches),'unmatched_count':len(pt)+len(onnx)-2*len(matches),'minimum_matched_iou':min((m['iou'] for m in matches),default=None),'maximum_confidence_abs_delta':max((m['confidence_abs_delta'] for m in matches),default=None),'maximum_bbox_abs_delta_px':max((m['bbox_max_abs_delta_px'] for m in matches),default=None),'matches':matches}

def records(result):
 out=[]
 if result.boxes is not None:
  for box in result.boxes:out.append({'class_id':int(box.cls[0].cpu()),'confidence':float(box.conf[0].cpu()),'bbox_xyxy':[float(v) for v in box.xyxy[0].cpu().tolist()]})
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,required=True);p.add_argument('--onnx',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--imgsz',type=int,default=640);p.add_argument('--confidence',type=float,default=.25);p.add_argument('--nms-iou',type=float,default=.45);p.add_argument('--minimum-box-iou',type=float,default=.8);a=p.parse_args()
 for f in (a.weights,a.onnx,a.image):
  if not f.is_file():p.error(f'file not found: {f}')
 import torch
 from ultralytics import YOLO
 image=Image.open(a.image).convert('RGB');pt=YOLO(str(a.weights));ox=YOLO(str(a.onnx))
 device='mps' if torch.backends.mps.is_available() else 'cpu'
 start=time.perf_counter();pr=pt.predict(image,imgsz=a.imgsz,conf=a.confidence,iou=a.nms_iou,device=device,verbose=False)[0];pt_ms=(time.perf_counter()-start)*1000
 start=time.perf_counter();orr=ox.predict(image,imgsz=a.imgsz,conf=a.confidence,iou=a.nms_iou,device='cpu',verbose=False)[0];onnx_ms=(time.perf_counter()-start)*1000
 paired=compare_rows(records(pr),records(orr),a.minimum_box_iou)
 paired.update({'status':'PASS' if paired['unmatched_count']==0 else 'MISMATCH','image':str(a.image.resolve()),'imgsz':a.imgsz,'confidence_threshold':a.confidence,'nms_iou':a.nms_iou,'minimum_box_iou_threshold':a.minimum_box_iou,'pytorch_device':device,'onnx_provider':'CPUExecutionProvider','pytorch_elapsed_ms':pt_ms,'onnx_first_call_elapsed_ms':onnx_ms,'note':'ONNX elapsed time includes first-call/session initialization if this is the first prediction.'})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(paired,indent=2)+'\n');print(json.dumps(paired,indent=2));return 0 if paired['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())

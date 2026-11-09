#!/usr/bin/env python3
"""Run an isolated research component detector (.pt or .onnx) and save its overlay."""
import argparse, json, time
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--confidence',type=float,default=.25);p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
    a=p.parse_args()
    if not a.model.is_file() or not a.image.is_file():p.error('model and image must exist')
    import torch
    from ultralytics import YOLO
    device=('cpu' if a.model.suffix.lower()=='.onnx' else ('mps' if torch.backends.mps.is_available() else 'cpu')) if a.device=='auto' else a.device
    if a.model.suffix.lower()=='.onnx':device='cpu'
    if device=='mps' and not torch.backends.mps.is_available():p.error('MPS unavailable')
    model=YOLO(str(a.model));image=Image.open(a.image).convert('RGB');start=time.perf_counter()
    result=model.predict(image,imgsz=a.imgsz,conf=a.confidence,device=device,verbose=False)[0]
    elapsed=(time.perf_counter()-start)*1000;names=result.names;detections=[]
    if result.boxes is not None:
        for box in result.boxes:
            cls=int(box.cls[0].cpu());detections.append({'class_id':cls,'class_name':str(names[cls]),'confidence':float(box.conf[0].cpu()),'bbox_xyxy':[float(x) for x in box.xyxy[0].cpu().tolist()]})
    draw=ImageDraw.Draw(image)
    for d in detections:
        x1,y1,x2,y2=d['bbox_xyxy'];draw.rectangle((x1,y1,x2,y2),outline=(30,255,90),width=max(2,image.width//900))
        draw.text((x1,max(0,y1-18)),f"{d['class_name']} {d['confidence']:.2f}",fill=(30,255,90))
    a.output_dir.mkdir(parents=True,exist_ok=True);overlay=a.output_dir/(a.image.stem+'_experimental.jpg');image.save(overlay,quality=92)
    report={'model_id':'EXPERIMENTAL_COMPONENT_DETECTOR','model_path':str(a.model.resolve()),'model_sha256':__import__('hashlib').sha256(a.model.read_bytes()).hexdigest(),'class_contract':list(names.values()) if isinstance(names,dict) else list(names),'image':str(a.image.resolve()),'device':device,'confidence_threshold':a.confidence,'inference_latency_ms':elapsed,'detections':detections,'overlay':str(overlay.resolve()),'production_model':False}
    (a.output_dir/(a.image.stem+'_experimental.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())

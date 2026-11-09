#!/usr/bin/env python3
"""Isolated YOLOv8/YOLO26 ONNX parity, head comparison, and latency benchmark."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, statistics, threading, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/component_training/manifests/pcb-dslr-part1-ic-manifest.json'
TOLERANCE={'minimum_detection_iou':.80,'maximum_confidence_abs_error':.05,'maximum_box_coordinate_error_px':4.0}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def image_rows(manifest_path=MANIFEST,split='validation'):
 m=json.loads(manifest_path.read_text())
 rows=[]
 for r in m['images']:
  if r['split']!=split:continue
  p=ROOT/r['image_path']
  with Image.open(p) as im:
   im.load(); image=im.convert('RGB')
  if list(image.size)!=r['image_size_px']:raise ValueError(f"dimension mismatch: {r['image_id']}")
  rows.append({'image_id':r['image_id'],'path':p,'image':image,'board_id':r['board_id'],'ground_truth':r['annotations']})
 if not rows:raise ValueError(f'no images for split {split}')
 return m,rows

def detections(result):
 rows=[]
 if result.boxes is not None:
  for b in result.boxes:
   rows.append({'class_id':int(b.cls[0].cpu()),'confidence':float(b.conf[0].cpu()),'bbox_xyxy':[float(x) for x in b.xyxy[0].cpu().tolist()]})
 return rows

def decode_nms_free_output(output,confidence=.25):
 """Decode YOLO26 one-to-one ONNX rows [x1,y1,x2,y2,score,class]. No NMS."""
 arr=np.asarray(output)
 if arr.size==0:return []
 if arr.ndim==3 and arr.shape[0]==1:arr=arr[0]
 if arr.ndim!=2 or arr.shape[1]!=6:raise ValueError(f'expected one-to-one output [N,6], got {arr.shape}')
 result=[]
 for x1,y1,x2,y2,score,cls in arr:
  if not np.isfinite([x1,y1,x2,y2,score,cls]).all():raise ValueError('non-finite one-to-one output')
  if score<confidence or x2<=x1 or y2<=y1:continue
  result.append({'class_id':int(round(float(cls))),'confidence':float(score),'bbox_xyxy':[float(x1),float(y1),float(x2),float(y2)]})
 return result

def box_iou(a,b):
 x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
 inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
 return inter/(aa+bb-inter) if aa+bb>inter else 0.0

def compare(a,b,minimum_iou=TOLERANCE['minimum_detection_iou']):
 unused=set(range(len(b)));matches=[];matched_left=set()
 for i,x in sorted(enumerate(a),key=lambda pair:-pair[1]['confidence']):
  opts=[(box_iou(x['bbox_xyxy'],b[j]['bbox_xyxy']),j) for j in unused if x['class_id']==b[j]['class_id']]
  iou,j=max(opts,default=(0,-1))
  if iou>=minimum_iou:
   unused.remove(j);matched_left.add(i);y=b[j]
   matches.append({'iou':float(iou),'confidence_abs_error':abs(x['confidence']-y['confidence']),'box_coordinate_max_abs_error_px':max(abs(u-v) for u,v in zip(x['bbox_xyxy'],y['bbox_xyxy']))})
 return {'left_count':len(a),'right_count':len(b),'matched_count':len(matches),'missing_or_extra_count':len(a)+len(b)-2*len(matches),
  'minimum_matched_iou':min((x['iou'] for x in matches),default=None),'maximum_confidence_abs_error':max((x['confidence_abs_error'] for x in matches),default=None),
  'maximum_box_coordinate_error_px':max((x['box_coordinate_max_abs_error_px'] for x in matches),default=None),'matches':matches,
  'unmatched_left':[{'class_id':x['class_id'],'confidence':x['confidence'],'bbox_xyxy':x['bbox_xyxy']} for i,x in enumerate(a) if i not in matched_left],
  'unmatched_right':[{'class_id':b[j]['class_id'],'confidence':b[j]['confidence'],'bbox_xyxy':b[j]['bbox_xyxy']} for j in sorted(unused)]}

def _matched_gt(gt,preds,threshold=.5):
 unused=set(range(len(gt)));matched=set();matched_pred=set()
 for p_i,p in sorted(enumerate(preds),key=lambda x:-x[1]['confidence']):
  opts=[(box_iou(p['bbox_xyxy'],g['bbox_xyxy']),i) for i,g in enumerate(gt) if i in unused]
  overlap,index=max(opts,default=(0,-1))
  if overlap>=threshold:unused.remove(index);matched.add(index);matched_pred.add(p_i)
 return matched,matched_pred

def render_cross_model(rows,v8,y26,out):
 out.mkdir(parents=True,exist_ok=True);v8_recovered=[];v26_recovered=[];counts=[]
 for row,a,b in zip(rows,v8,y26):
  gt=row['ground_truth'];ma,_=_matched_gt(gt,a);mb,_=_matched_gt(gt,b)
  v8_recovered.extend(f"{row['image_id']}:{i}" for i in mb-ma);v26_recovered.extend(f"{row['image_id']}:{i}" for i in ma-mb)
  canvas=Image.new('RGB',(row['image'].width*2,row['image'].height+28),(25,25,25));draw=ImageDraw.Draw(canvas)
  for j,(name,preds) in enumerate((('YOLOv8s',a),('YOLO26s',b))):
   xoff=j*row['image'].width;canvas.paste(row['image'],(xoff,28));draw.text((xoff+6,6),name,fill='white')
   _,true_positive=_matched_gt(gt,preds)
   for g in gt:
    x1,y1,x2,y2=g['bbox_xyxy'];draw.rectangle((xoff+x1,y1+28,xoff+x2,y2+28),outline=(40,150,255),width=2)
   for k,p in enumerate(preds):
    x1,y1,x2,y2=p['bbox_xyxy'];color=(30,235,70) if k in true_positive else (255,135,35)
    draw.rectangle((xoff+x1,y1+28,xoff+x2,y2+28),outline=color,width=2)
  canvas.save(out/(row['image_id'].replace('/','_').replace(':','_')+'_heads.jpg'),quality=88)
  counts.append({'image_id':row['image_id'],'ground_truth':len(gt),'yolov8s_detected':len(ma),'yolo26s_detected':len(mb),'yolov8s_fp':len(a)-len(ma),'yolo26s_fp':len(b)-len(mb)})
 return {'per_image':counts,'gt_missed_by_yolov8_recovered_by_yolo26':v8_recovered,'gt_missed_by_yolo26_recovered_by_yolov8':v26_recovered,'overlay_dir':str(out.resolve())}

def check_contract(model):
 names=model.names
 names=list(names.values()) if isinstance(names,dict) else list(names)
 if names!=['integrated_circuit']:raise ValueError(f'expected class contract integrated_circuit, got {names}')
 return names

def model_summary(weights):
 from ultralytics import YOLO
 from ultralytics.utils.torch_utils import get_flops, get_num_params
 model=YOLO(str(weights))
 check_contract(model)
 head=model.model.model[-1]
 raw_flops=float(get_flops(model.model,640));end2end_flops=None
 if hasattr(head,'one2one_cv2'):
  old=head.end2end
  try:head.end2end=True;end2end_flops=float(get_flops(model.model,640))
  finally:head.end2end=old
 return {'architecture':str(model.model.yaml.get('yaml_file',weights.stem)),'parameters_including_all_heads':int(get_num_params(model.model)),
  'gflops_one_to_many_640':raw_flops,'gflops_one_to_one_640':end2end_flops,'checkpoint_size_bytes':weights.stat().st_size,'checkpoint_sha256':sha(weights)}

def export_onnx(weights,output,nms):
 import onnx, onnxruntime as ort
 from ultralytics import YOLO, __version__
 output.parent.mkdir(parents=True,exist_ok=True)
 model=YOLO(str(weights));check_contract(model)
 generated=Path(model.export(format='onnx',imgsz=640,opset=17,nms=nms,device='cpu',simplify=False))
 if generated.resolve()!=output.resolve():shutil.copy2(generated,output)
 onnx.checker.check_model(str(output))
 session=ort.InferenceSession(str(output),providers=['CPUExecutionProvider'])
 output_info=[{'name':x.name,'shape':x.shape,'type':x.type} for x in session.get_outputs()]
 one_to_one_contract=None
 if nms is False:
  shape=session.get_inputs()[0].shape
  if all(isinstance(x,int) for x in shape):
   dummy=session.run(None,{session.get_inputs()[0].name:np.zeros(shape,dtype=np.float32)})[0]
   one_to_one_contract={'decoded_dummy_detection_count':len(decode_nms_free_output(dummy,.25)),'adapter':'decode_nms_free_output','applies_nms':False}
 return {'weights':str(weights.resolve()),'weights_sha256':sha(weights),'onnx':str(output.resolve()),'onnx_sha256':sha(output),'onnx_size_bytes':output.stat().st_size,
  'ultralytics':__version__,'requested_nms':nms,'onnx_inputs':[{'name':x.name,'shape':x.shape,'type':x.type} for x in session.get_inputs()],
  'onnx_outputs':output_info,'one_to_one_adapter':one_to_one_contract,'providers':session.get_providers()}

def predict_rows(model_path,rows,head,device,conf=.25,iou=.45):
 import torch
 from ultralytics import YOLO
 model=YOLO(str(model_path));check_contract(model);results=[]
 if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
 for r in rows:
  kwargs={'imgsz':640,'conf':conf,'device':device,'verbose':False}
  if head=='yolo26_nms_free':kwargs['nms']=False
  else:kwargs['iou']=iou
  pred=model.predict(r['image'],**kwargs)[0]
  results.append(detections(pred))
 return results

def parity_for_paths(weights,raw_onnx,nmsfree_onnx,rows,model_id,output_dir):
 pt=predict_rows(weights,rows,'raw','mps')
 pt_cpu=predict_rows(weights,rows,'raw','cpu')
 on=predict_rows(raw_onnx,rows,'raw','cpu')
 per_image=[{'image_id':r['image_id'],**compare(x,y)} for r,x,y in zip(rows,pt,on)]
 per_image_cpu=[{'image_id':r['image_id'],**compare(x,y)} for r,x,y in zip(rows,pt_cpu,on)]
 raw={'model_id':model_id,'path':'raw_one_to_many_external_nms','thresholds':{'confidence':.25,'nms_iou':.45},'tolerance':TOLERANCE,
  'pytorch_mps_vs_onnx_cpu':per_image,'pytorch_cpu_vs_onnx_cpu':per_image_cpu}
 if nmsfree_onnx:
  one_pt=predict_rows(weights,rows,'yolo26_nms_free','mps')
  one_pt_cpu=predict_rows(weights,rows,'yolo26_nms_free','cpu')
  one_onnx=predict_rows(nmsfree_onnx,rows,'yolo26_nms_free','cpu')
  raw['nms_free']={'path':'one_to_one_nms_free','thresholds':{'confidence':.25,'iou':None},'tolerance':TOLERANCE,
   'pytorch_mps_vs_onnx_cpu':[{'image_id':r['image_id'],**compare(x,y)} for r,x,y in zip(rows,one_pt,one_onnx)],
   'pytorch_cpu_vs_onnx_cpu':[{'image_id':r['image_id'],**compare(x,y)} for r,x,y in zip(rows,one_pt_cpu,one_onnx)]}
  raw['head_comparison']=[{'image_id':r['image_id'],**compare(x,y,minimum_iou=.5)} for r,x,y in zip(rows,pt,one_pt)]
 output_dir.mkdir(parents=True,exist_ok=True)
 (output_dir/'onnx-parity.json').write_text(json.dumps(raw,indent=2)+'\n')
 return raw

def _sync(device):
 if device=='mps':
  import torch;torch.mps.synchronize()

def benchmark(model_path,rows,head,device,repeats=100,warmup=10,conf=.25,iou=.45):
 import torch, psutil
 from ultralytics import YOLO
 model=YOLO(str(model_path));check_contract(model)
 kwargs={'imgsz':640,'conf':conf,'device':device,'verbose':False}
 if head=='yolo26_nms_free':kwargs['nms']=False
 else:kwargs['iou']=iou
 process=psutil.Process();peak={'rss':process.memory_info().rss};stop=threading.Event()
 def sample_memory():
  while not stop.is_set():
   peak['rss']=max(peak['rss'],process.memory_info().rss);stop.wait(.01)
 monitor=threading.Thread(target=sample_memory,daemon=True);monitor.start()
 first0=time.perf_counter();model.predict(rows[0]['image'],**kwargs)[0];_sync(device);first=(time.perf_counter()-first0)*1000
 predictor=model.predictor;phase={'preprocess':[],'inference':[],'postprocess':[]};outer=[]
 for name in phase:
  original=getattr(predictor,name)
  def wrapped(*args,_name=name,_original=original,**kw):
   t=time.perf_counter();result=_original(*args,**kw);_sync(device);phase[_name].append((time.perf_counter()-t)*1000);return result
  setattr(predictor,name,wrapped)
 for j in range(warmup):model.predict(rows[j%len(rows)]['image'],**kwargs)[0]
 for values in phase.values():values.clear()
 for j in range(repeats):
  _sync(device);t=time.perf_counter();model.predict(rows[j%len(rows)]['image'],**kwargs)[0];_sync(device);outer.append((time.perf_counter()-t)*1000)
 peak['rss']=max(peak['rss'],process.memory_info().rss);stop.set();monitor.join()
 def stats(v):return {'median_ms':statistics.median(v),'p95_ms':float(np.percentile(v,95)),'mean_ms':statistics.mean(v),'count':len(v)}
 return {'model':str(model_path.resolve()),'model_sha256':sha(model_path) if model_path.suffix=='.pt' else None,'head':head,'device':device,'backend':'ONNX Runtime CPU' if model_path.suffix=='.onnx' else 'PyTorch MPS',
  'imgsz':640,'batch':1,'confidence_threshold':conf,'nms_iou':iou if head!='yolo26_nms_free' else None,'warmup_calls':warmup,'timed_calls':repeats,'first_call_ms':first,
  'end_to_end':{**stats(outer),'images_per_second':1000/statistics.mean(outer)},'phase':{k:stats(v) for k,v in phase.items()},'peak_process_rss_bytes':peak['rss'],'model_size_bytes':model_path.stat().st_size}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--yolov8',type=Path,required=True);p.add_argument('--yolo26',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--repeats',type=int,default=100);p.add_argument('--manifest',type=Path,default=MANIFEST)
 a=p.parse_args();manifest,rows=image_rows(a.manifest,'validation');out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
 exports={}
 exports['yolov8_raw']=export_onnx(a.yolov8,out/'yolov8s-raw.onnx',nms=None)
 exports['yolo26_raw']=export_onnx(a.yolo26,out/'yolo26s-raw.onnx',nms=None)
 exports['yolo26_nms_free']=export_onnx(a.yolo26,out/'yolo26s-nms-free.onnx',nms=False)
 parity={'yolov8s':parity_for_paths(a.yolov8,out/'yolov8s-raw.onnx',None,rows,'YOLOv8s',out/'parity/yolov8s'),
  'yolo26s':parity_for_paths(a.yolo26,out/'yolo26s-raw.onnx',out/'yolo26s-nms-free.onnx',rows,'YOLO26s',out/'parity/yolo26s')}
 overlay_predictions={'yolov8s':predict_rows(a.yolov8,rows,'raw','mps'),'yolo26s_raw':predict_rows(a.yolo26,rows,'raw','mps')}
 head_overlay=render_cross_model(rows,overlay_predictions['yolov8s'],overlay_predictions['yolo26s_raw'],out/'same-image-head-overlays')
 paths=[('yolov8s_raw_pt',a.yolov8,'raw','mps'),('yolov8s_raw_onnx',out/'yolov8s-raw.onnx','raw','cpu'),('yolo26s_raw_pt',a.yolo26,'raw','mps'),('yolo26s_raw_onnx',out/'yolo26s-raw.onnx','raw','cpu'),('yolo26s_nmsfree_pt',a.yolo26,'yolo26_nms_free','mps'),('yolo26s_nmsfree_onnx',out/'yolo26s-nms-free.onnx','yolo26_nms_free','cpu')]
 perf={name:benchmark(path,rows,head,device,a.repeats) for name,path,head,device in paths}
 parity_summary={}
 for model_id,result in parity.items():
  def within(x):
   return (x['missing_or_extra_count']==0 and (x['minimum_matched_iou'] is None or x['minimum_matched_iou']>=TOLERANCE['minimum_detection_iou'])
    and (x['maximum_confidence_abs_error'] is None or x['maximum_confidence_abs_error']<=TOLERANCE['maximum_confidence_abs_error'])
    and (x['maximum_box_coordinate_error_px'] is None or x['maximum_box_coordinate_error_px']<=TOLERANCE['maximum_box_coordinate_error_px']))
  parity_summary[model_id]={'raw_mps_vs_onnx_cpu':{'image_count':len(result['pytorch_mps_vs_onnx_cpu']),'all_detections_within_tolerance':all(within(x) for x in result['pytorch_mps_vs_onnx_cpu'])},
   'raw_cpu_vs_onnx_cpu':{'image_count':len(result['pytorch_cpu_vs_onnx_cpu']),'all_detections_within_tolerance':all(within(x) for x in result['pytorch_cpu_vs_onnx_cpu'])}}
  if 'nms_free' in result:
   parity_summary[model_id]['nmsfree_mps_vs_onnx_cpu']={'image_count':len(result['nms_free']['pytorch_mps_vs_onnx_cpu']),'all_detections_within_tolerance':all(within(x) for x in result['nms_free']['pytorch_mps_vs_onnx_cpu'])}
   parity_summary[model_id]['nmsfree_cpu_vs_onnx_cpu']={'image_count':len(result['nms_free']['pytorch_cpu_vs_onnx_cpu']),'all_detections_within_tolerance':all(within(x) for x in result['nms_free']['pytorch_cpu_vs_onnx_cpu'])}
 report={'schema_version':'circuitcare.phase21-3-2-e-onnx-benchmark.v1','dataset_manifest_sha256':sha(a.manifest),'dataset_split':'validation','previously_exposed_test_used':False,'image_count':len(rows),'board_count':len({r['board_id'] for r in rows}),
  'model_summary':{'yolov8s':model_summary(a.yolov8),'yolo26s':model_summary(a.yolo26)},
  'fixed_tolerance':TOLERANCE,'exports':exports,'parity':parity_summary,'same_image_head_overlay_summary':head_overlay,
  'detail_paths':{'yolov8s':str((out/'parity/yolov8s/onnx-parity.json').resolve()),'yolo26s':str((out/'parity/yolo26s/onnx-parity.json').resolve())},'performance':perf}
 (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())

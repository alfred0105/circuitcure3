#!/usr/bin/env python3
"""Download direct CAD/board/schematic assets linked from catalog product pages."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import requests

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PAGES=ROOT/"data/component_training/raw/external/pcb_photo_cad_pairs/catalog_product_pages_cad_links/page_image_manifest.json"
DEFAULT_OUTPUT=ROOT/"data/component_training/raw/external/pcb_photo_cad_pairs/catalog_direct_cad_assets"
CAD_SUFFIXES={".zip",".brd",".sch",".kicad_pcb",".kicad_sch",".pcbdoc",".schdoc",".gbr",".ger",".drl",".csv",".xml",".pdf"}
EXTRACT_SUFFIXES={".brd",".sch",".kicad_pcb",".kicad_sch",".pcbdoc",".schdoc",".gbr",".ger",".gtl",".gbl",".gts",".gbs",".gto",".gbo",".gko",".drl",".csv",".xml"}


def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def is_cad(url:str)->bool:
    p=urlparse(url)
    path=unquote(p.path).lower()
    suffix=Path(path).suffix.lower()
    if suffix in CAD_SUFFIXES:
        if suffix==".pdf" and not re.search(r"schem|circuit|board|hardware|pcb",path): return False
        return True
    return False


def download(task, out:Path, max_bytes:int)->dict:
    url,candidate_ids,source_page=task
    result={"url":url,"candidate_ids":candidate_ids,"source_page":source_page}
    try:
        r=requests.get(url,headers={"User-Agent":"CircuitCare-Internal-PCB-CAD-Curator/1.0"},timeout=(15,60),stream=True,allow_redirects=True)
        r.raise_for_status(); chunks=[];size=0
        for c in r.iter_content(1024*1024):
            if not c:continue
            size+=len(c)
            if size>max_bytes:return {**result,"status":"SKIPPED_FILE_TOO_LARGE","content_length":r.headers.get("content-length")}
            chunks.append(c)
        data=b"".join(chunks)
        if not data:return {**result,"status":"EMPTY_RESPONSE"}
        digest=sha(data); basename=Path(unquote(urlparse(r.url).path)).name or "download.bin"
        suffix=Path(basename).suffix or ".bin"
        target=out/"files"/candidate_ids[0]/(digest+"__"+basename)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        extracted=[]
        if zipfile.is_zipfile(target):
            extract_dir=out/"extracted"/candidate_ids[0]/digest
            with zipfile.ZipFile(target) as zf:
                for member in zf.infolist():
                    if member.is_dir():continue
                    rel=PurePosixPath(member.filename)
                    if rel.is_absolute() or ".." in rel.parts:continue
                    if Path(rel.name).suffix.lower() not in EXTRACT_SUFFIXES:continue
                    dest=extract_dir/Path(*rel.parts)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(zf.read(member))
                    extracted.append({"path":str(dest.relative_to(out)),"member":member.filename,"sha256":sha(dest.read_bytes()),"size_bytes":dest.stat().st_size})
        return {**result,"status":"DOWNLOADED","final_url":r.url,"path":str(target.relative_to(out)),"sha256":digest,"size_bytes":len(data),"content_type":r.headers.get("content-type"),"extracted_cad_count":len(extracted),"extracted_cad":extracted}
    except Exception as exc:
        return {**result,"status":"DOWNLOAD_FAILED","error":str(exc)}


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--pages",type=Path,default=DEFAULT_PAGES);p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--workers",type=int,default=8);p.add_argument("--max-mb",type=int,default=120);p.add_argument("--append-existing",action="store_true")
    a=p.parse_args();pages=json.loads(a.pages.read_text(encoding="utf-8")); grouped={}
    for row in pages["pages"]:
        for url in row.get("cad_link_candidates",[]):
            if not is_cad(url):continue
            if url not in grouped:grouped[url]={"candidate_ids":set(),"source_page":row["page_url"]}
            grouped[url]["candidate_ids"].update(row.get("candidate_ids",[]))
    a.output.mkdir(parents=True,exist_ok=True);manifest_path=a.output/"direct_cad_manifest.json"
    old=[]
    if a.append_existing and manifest_path.is_file(): old=json.loads(manifest_path.read_text(encoding="utf-8")).get("items",[])
    old_by_url={x["url"]:x for x in old}
    tasks=[]
    for url,v in grouped.items():
        if url in old_by_url:
            old_by_url[url]["candidate_ids"]=sorted(set(old_by_url[url].get("candidate_ids",[]))|v["candidate_ids"])
            continue
        tasks.append((url,sorted(v["candidate_ids"]),v["source_page"]))
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(download,t,a.output,a.max_mb*1024*1024):t[0] for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            try:r=f.result()
            except Exception as e:r={"url":futures[f],"status":"ERROR","error":str(e)}
            results.append(r);print(f"[{i}/{len(tasks)}] {r['status']} {r['url']}",flush=True)
    all_items=list(old_by_url.values())+results
    manifest={"schema_version":"circuitcare.direct-product-page-cad-assets.v1","source_page_manifests":sorted(set([str(a.pages)]+[x.get("source_page_manifest","") for x in old if x.get("source_page_manifest")])),"direct_asset_url_count":len(all_items),"downloaded_count":sum(x.get("status")=="DOWNLOADED" for x in all_items),"extracted_cad_file_count":sum(x.get("extracted_cad_count",0) for x in all_items),"items":sorted(all_items,key=lambda x:x["url"])}
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("direct_asset_url_count","downloaded_count","extracted_cad_file_count")},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())

#!/usr/bin/env python3
"""Collect image assets linked from non-GitHub product/documentation photo pages."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/component_training/manifests/pcb-photo-cad-source-catalog.json"
DEFAULT_OUTPUT = ROOT / "data/component_training/raw/external/pcb_photo_cad_pairs/catalog_product_page_images"
LOCK = threading.Lock()
SKIP = ("logo", "icon", "avatar", "spinner", "pixel", "tracking", "social", "favicon", "flag", "placeholder")


class Images(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls=[]; self.anchors=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag == "a" and a.get("href"):
            self.anchors.append(a["href"])
        elif tag == "meta" and a.get("property", "").lower() in {"og:image", "og:image:url", "twitter:image"}:
            if a.get("content"): self.urls.append((a["content"], "social_meta"))
        elif tag == "link" and "image_src" in a.get("rel", "").lower():
            if a.get("href"): self.urls.append((a["href"], "link_image_src"))
        elif tag == "img":
            for key in ("src", "data-src", "data-original", "data-zoom-image", "data-lazy-src"):
                if a.get(key): self.urls.append((a[key], "img_"+key))
            for entry in a.get("srcset", "").split(","):
                value=entry.strip().split()
                if value: self.urls.append((value[0], "img_srcset"))
        elif tag == "source" and a.get("srcset"):
            for entry in a["srcset"].split(","):
                value=entry.strip().split()
                if value: self.urls.append((value[0], "source_srcset"))


def file_hash(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def fetch_page(url: str, candidate_ids: list[str], output: Path, links_only: bool=False) -> dict:
    result={"page_url":url,"candidate_ids":candidate_ids,"status":"FETCHED"}
    try:
        response=requests.get(url,headers={"User-Agent":"CircuitCare-Internal-PCB-Photo-Curator/1.0"},timeout=(15,40),allow_redirects=True)
        response.raise_for_status()
        if "html" not in response.headers.get("content-type","").lower():
            return {**result,"status":"NON_HTML_SOURCE","final_url":response.url,"content_type":response.headers.get("content-type")}
        parser=Images(); parser.feed(response.text)
        cad_links=[]
        for raw in parser.anchors:
            link=urljoin(response.url,raw)
            if re.search(r"(?i)(\.kicad_(pcb|sch)|\.brd(?:$|\?)|\.sch(?:$|\?)|\.pcbdoc|\.schdoc|\.eagle|\.zip(?:$|\?)|gerber|schematic|\bpcb\b|hardware|open.?source.?hardware|cad.?files|design.?files)",link):
                if link not in cad_links: cad_links.append(link)
        if links_only:
            return {**result,"final_url":response.url,"http_status":response.status_code,"cad_link_candidates":cad_links,"status":"LINKS_DISCOVERED"}
        candidates=[]
        seen=set()
        for raw,found_by in parser.urls:
            image_url=urljoin(response.url,raw)
            parsed=urlparse(image_url)
            if parsed.scheme not in {"http","https"} or image_url in seen: continue
            if any(s in parsed.path.lower() for s in SKIP): continue
            ext=Path(unquote(parsed.path)).suffix.lower()
            if ext in {".svg",".gif",".ico"}: continue
            seen.add(image_url); candidates.append((image_url,found_by))
        saved=[]
        for image_url,found_by in candidates:
            try:
                img=requests.get(image_url,headers={"User-Agent":"CircuitCare-Internal-PCB-Photo-Curator/1.0"},timeout=(15,40),stream=True,allow_redirects=True)
                img.raise_for_status()
                ctype=img.headers.get("content-type","").lower()
                if not ctype.startswith("image/") or "svg" in ctype or "gif" in ctype: continue
                chunks=[]; size=0
                for chunk in img.iter_content(1024*1024):
                    if not chunk: continue
                    size+=len(chunk)
                    if size>40*1024*1024: break
                    chunks.append(chunk)
                if size>40*1024*1024: continue
                data=b"".join(chunks)
                if len(data)<4096: continue
                digest=file_hash(data)
                ext=Path(unquote(urlparse(img.url).path)).suffix.lower()
                if ext not in {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}:
                    ext={"image/jpeg":".jpg","image/png":".png","image/webp":".webp","image/avif":".avif"}.get(ctype,".img")
                target=output/"images"/candidate_ids[0]/(digest+ext)
                target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
                dims=None
                try:
                    with Image.open(target) as im: dims=list(im.size)
                except Exception: pass
                saved.append({"path":str(target.relative_to(output)),"source_image_url":img.url,"sha256":digest,"bytes":len(data),"content_type":ctype,"dimensions_px":dims,"discovered_by":found_by,"role":"PRODUCT_PAGE_IMAGE_CANDIDATE_NOT_VISUALLY_CLASSIFIED"})
            except Exception:
                continue
        result.update({"final_url":response.url,"http_status":response.status_code,"discovered_image_url_count":len(candidates),"saved_image_count":len(saved),"images":saved,"cad_link_candidates":cad_links})
        if not saved: result["status"]="NO_DOWNLOADABLE_IMAGES_FOUND"
        return result
    except Exception as exc:
        return {**result,"status":"PAGE_FETCH_FAILED","error":str(exc)}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--catalog",type=Path,default=DEFAULT_CATALOG);p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--workers",type=int,default=8);p.add_argument("--links-only",action="store_true");p.add_argument("--include-cad-links",action="store_true",help="Also fetch non-GitHub product/design source pages listed as CAD links")
    a=p.parse_args();catalog=json.loads(a.catalog.read_text(encoding="utf-8"));page_to_ids={}
    for row in catalog["candidates"]:
        sources = list(row.get("photo_or_product_links", []))
        if a.include_cad_links:
            sources += list(row.get("cad_or_design_links", []))
        for item in sources:
            url=item["url"]
            if "github.com/" in url: continue
            if url not in page_to_ids: page_to_ids[url]=set()
            page_to_ids[url].add(row["candidate_id"])
    a.output.mkdir(parents=True,exist_ok=True); results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(fetch_page,url,sorted(ids),a.output,a.links_only):url for url,ids in page_to_ids.items()}
        for i,f in enumerate(as_completed(futures),1):
            try:r=f.result()
            except Exception as exc:r={"page_url":futures[f],"status":"ERROR","error":str(exc)}
            results.append(r)
            with LOCK:
                print(f"[{i}/{len(futures)}] {r['page_url']} {r['status']} images={r.get('saved_image_count',0)}",flush=True)
    summary={"schema_version":"circuitcare.product-page-photo-assets.v1","source_catalog_sha256":catalog["source_sha256"],"page_count":len(results),"downloaded_image_count":sum(x.get("saved_image_count",0) for x in results),"pages":sorted(results,key=lambda x:x["page_url"])}
    (a.output/"page_image_manifest.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"page_count":summary["page_count"],"downloaded_image_count":summary["downloaded_image_count"],"output":str(a.output)},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())

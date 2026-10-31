"""Fetch public originals with checksums; large files stay outside Git."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def md5(path):
    h=hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def download(url,path,expected=None):
    if path.exists() and (expected is None or md5(path)==expected):
        print("Existing verified",path.name,flush=True); return
    tmp=path.with_suffix(path.suffix+".part")
    with urllib.request.urlopen(url,timeout=120) as response,tmp.open("wb") as stream:
        shutil.copyfileobj(response,stream,8*1024*1024)
    if expected and md5(tmp)!=expected:
        raise RuntimeError(f"Checksum mismatch for {path.name}; original not replaced")
    tmp.replace(path); print("Downloaded",path.name,flush=True)


def main():
    raw=ROOT / "data/raw/der_fault"; raw.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen("https://zenodo.org/api/records/22904220",timeout=60) as response: record=json.load(response)
    (raw / "zenodo_record.json").write_text(json.dumps(record,indent=2))
    for f in record["files"]: download(f["links"]["self"],raw / f["key"],f["checksum"].split(":",1)[1])
    iowa=ROOT / "data/raw/iowa"; iowa.mkdir(parents=True,exist_ok=True)
    url="https://wzy.ece.iastate.edu/publication/OpenDSS%20Model.zip"
    archive=iowa / "OpenDSS_Model.zip"; download(url,archive)
    destination=(iowa / "model").resolve(); destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target=(destination / info.filename).resolve()
            if not target.is_relative_to(destination): raise ValueError("Unsafe archive path")
        z.extractall(destination)
    (ROOT / "reports/source_topology_provenance.json").write_text(json.dumps({"source":url,"bytes":archive.stat().st_size,"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"note":"Source circuit; exact study-specific modifications not distributed."},indent=2))


if __name__=="__main__": main()

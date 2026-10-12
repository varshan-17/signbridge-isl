"""Fetch only the 26 CC0 visualization photos from RealSign's ZIP via HTTP ranges."""
import io, json, sys, urllib.request, zipfile
from pathlib import Path

URL = 'https://media.githubusercontent.com/media/RealSign62/RealSign-Indian-Sign-Language-Dataset/17c51dcc158b7b6359b7c6667edc51c7f148f3f4/Dataset.zip'
SIZE = 656689688

class RemoteZip(io.RawIOBase):
    def __init__(self): self.position = 0; self.cache = {}
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else SIZE+offset
        return self.position
    def read(self, size=-1):
        size = SIZE-self.position if size < 0 else min(size,SIZE-self.position)
        if size <= 0: return b''
        start=self.position; end=self.position+size
        for a,data in self.cache.items():
            if a<=start and end<=a+len(data):
                self.position=end
                return data[start-a:end-a]
        chunk_start=start if size>100_000 else (start//262144)*262144
        chunk_end=min(SIZE,max(end,chunk_start+262144))
        req = urllib.request.Request(URL, headers={'Range': f'bytes={chunk_start}-{chunk_end-1}'})
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 206: raise ValueError('Server must support partial downloads')
            data=response.read()
        if len(data)!=chunk_end-chunk_start: raise ValueError('Incomplete range')
        self.cache[chunk_start]=data
        self.position=end
        return data[start-chunk_start:end-chunk_start]

out=Path('public/references/letters')
out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(RemoteZip()) as archive:
    Path('tmp').mkdir(exist_ok=True)
    Path('tmp/reference-archive.json').write_text(json.dumps([f.filename for f in archive.infolist()]))
    entries=[f for f in archive.infolist() if 'letters' in f.filename.lower() and not f.is_dir()]
    if '--training' in sys.argv:
        entries=[]
        for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            samples=sorted([f for f in archive.infolist() if f.filename.startswith('Training/'+label+'/') and not f.is_dir()],key=lambda f:f.filename)
            entries.extend(samples[round(i*(len(samples)-1)/7)] for i in range(8))
        manifest=[]
        for entry in entries:
            label=Path(entry.filename).parent.name
            destination=Path('tmp/reference-training',label,Path(entry.filename).name)
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.read(entry))
            manifest.append({'label':label,'url':'/'+destination.as_posix(),'source':entry.filename})
            print('Saved',entry.filename,flush=True)
        Path('tmp/reference-training.json').write_text(json.dumps(manifest))
        sys.exit(0)
    print('Visualization entries:',[f.filename for f in entries],flush=True)
    for entry in entries:
        name=Path(entry.filename).name
        label=Path(name).stem.upper()
        if label not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' or len(label)!=1: continue
        if entry.file_size>2_000_000: raise ValueError('Unexpected reference size')
        (out/(label+Path(name).suffix.lower())).write_bytes(archive.read(entry))
        print('Saved',label,flush=True)

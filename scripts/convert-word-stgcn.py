"""Convert the released OpenHands INCLUDE ST-GCN checkpoint to browser ONNX.

Architecture adapted from AI4Bharat/OpenHands (Apache-2.0). The adjacency,
normalization, graph/temporal convolutions and classifier are loaded unchanged.
"""
import io,sys,json,hashlib,zipfile
from pathlib import Path
sys.path.insert(0,str(Path('tmp/model-conversion').resolve()))
import torch
from torch import nn
import torch.nn.functional as F

class CallbackMetadata: pass
torch.serialization.add_safe_globals([(CallbackMetadata,'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'),(CallbackMetadata,'pytorch_lightning.callbacks.early_stopping.EarlyStopping')])
with zipfile.ZipFile('tmp/model-upgrade/include_stgcn.zip') as archive:
    raw=archive.read(next(f for f in archive.namelist() if f.endswith('.ckpt')))
bundle=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
state={k.removeprefix('model.'):v for k,v in bundle['state_dict'].items()}

class GraphConv(nn.Module):
    def __init__(self,inputs,outputs,kernels):
        super().__init__(); self.kernels=kernels
        self.conv=nn.Conv2d(inputs,outputs*kernels,kernel_size=(1,1))
    def forward(self,x,a):
        x=self.conv(x);n,kc,t,v=x.shape
        return torch.einsum('nkctv,kvw->nctw',x.view(n,self.kernels,kc//self.kernels,t,v),a)

class Block(nn.Module):
    def __init__(self,inputs,outputs,kernels,stride=1,residual=True):
        super().__init__()
        self.gcn=GraphConv(inputs,outputs,kernels)
        self.tcn=nn.Sequential(nn.BatchNorm2d(outputs),nn.ReLU(),nn.Conv2d(outputs,outputs,(9,1),(stride,1),(4,0)),nn.BatchNorm2d(outputs),nn.Dropout(0))
        self.residual=(lambda x:0) if not residual else (lambda x:x) if inputs==outputs and stride==1 else nn.Sequential(nn.Conv2d(inputs,outputs,1,stride=(stride,1)),nn.BatchNorm2d(outputs))
    def forward(self,x,a): return F.relu(self.tcn(self.gcn(x,a))+self.residual(x))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('A',state['encoder.A'].clone())
        self.data_bn=nn.BatchNorm1d(54)
        channels=[2,64,64,64,64,128,128,128,256,256,256]
        self.st_gcn_networks=nn.ModuleList([Block(channels[i],channels[i+1],2,stride=2 if i in [4,7] else 1,residual=i!=0) for i in range(10)])
        self.edge_importance=nn.ParameterList([nn.Parameter(torch.ones_like(self.A)) for _ in range(10)])
    def forward(self,x):
        n,c,t,v=x.shape
        x=self.data_bn(x.permute(0,3,1,2).contiguous().view(n,v*c,t))
        x=x.view(n,v,c,t).permute(0,2,3,1).contiguous()
        for block,importance in zip(self.st_gcn_networks,self.edge_importance): x=block(x,self.A*importance)
        return F.avg_pool2d(x,x.shape[2:]).view(n,-1)

class Decoder(nn.Module):
    def __init__(self): super().__init__();self.classifier=nn.Linear(256,263)
    def forward(self,x): return self.classifier(x)
class Model(nn.Module):
    def __init__(self): super().__init__();self.encoder=Encoder();self.decoder=Decoder()
    def forward(self,poses): return self.decoder(self.encoder(poses))

torch.set_num_threads(1)
model=Model();model.load_state_dict(state,strict=True);model.eval()
meta=json.loads(Path('public/models/isl-words.json').read_text())
sample=torch.randn(1,2,120,27,generator=torch.Generator().manual_seed(42))
output=Path('public/models/isl-words.onnx')
torch.onnx.export(model,sample,str(output),input_names=['poses'],output_names=['logits'],opset_version=17,dynamo=False)
meta.update(architecture='ST-GCN',shape=[1,2,120,27],checkpointSha256=hashlib.sha256(raw).hexdigest(),onnxSha256=hashlib.sha256(output.read_bytes()).hexdigest(),checkpointAsset='include_stgcn.zip')
Path('public/models/isl-words.json').write_text(json.dumps(meta,indent=2))
with torch.no_grad(): expected=model(sample).flatten().tolist()
Path('tests/fixtures/word-parity.json').write_text(json.dumps({'shape':[1,2,120,27],'input':sample.flatten().tolist(),'logits':expected}))
print('Exported ST-GCN:',output.stat().st_size,'bytes',flush=True)
if Path('tmp/word-clips/features.json').exists():
    for clip in json.loads(Path('tmp/word-clips/features.json').read_text()):
        frames=torch.tensor(clip['raw'],dtype=torch.float32)
        center=((frames[:,3]+frames[:,4])/2).mean(0)
        scale=torch.linalg.vector_norm(frames[:,3]-frames[:,4],dim=-1).mean()
        for count in [48,120,180,240]:
            indexes=torch.linspace(0,len(frames)-1,count).round().long()
            poses=((frames[indexes]-center)/scale).permute(2,0,1).unsqueeze(0)
            with torch.no_grad(): values=torch.softmax(model(poses).flatten(),dim=0);top=torch.topk(values,3)
            print(clip['expected'],count,[(meta['classes'][i],round(float(v),3)) for i,v in zip(top.indices,top.values)],flush=True)

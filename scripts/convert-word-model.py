"""Export AI4Bharat's released INCLUDE LSTM without installing OpenHands.

Architecture follows OpenHands Network, PoseFlattener, RNNClassifier and
AttentionBlock (Apache-2.0). Preserve its published tensor convention rather
than silently changing the released checkpoint's forward pass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('tmp/model-conversion').resolve()))
import json
import csv
import hashlib
import zipfile
import torch
from torch import nn

for archive,target in [('include_lstm.zip','include_lstm'),('include_metadata.zip','include_metadata')]:
    destination=Path('tmp/models',target).resolve()
    if not destination.exists():
        with zipfile.ZipFile(Path('tmp/models',archive)) as z:
            for item in z.infolist():
                if not (destination/item.filename).resolve().is_relative_to(destination):
                    raise ValueError('Unsafe archive member')
            z.extractall(destination)
checkpoint = next(Path('tmp/models/include_lstm').rglob('*.ckpt'))
# The training checkpoint stores callback classes as dictionary keys. Inert
# placeholders let the restricted weights-only loader read those keys without
# importing or executing Lightning training callbacks.
class CallbackMetadata:
    pass
torch.serialization.add_safe_globals([
    (CallbackMetadata,'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'),
    (CallbackMetadata,'pytorch_lightning.callbacks.early_stopping.EarlyStopping'),
])
bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
state = bundle['state_dict']
print('Checkpoint tensors:', [(k,tuple(v.shape)) for k,v in state.items()])

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256,256,bias=False)
        self.fc2 = nn.Linear(512,256,bias=False)
    def forward(self,x):
        last=x[:,-1,:]
        scores=torch.bmm(self.fc1(x),last.unsqueeze(2)).squeeze(2)
        context=torch.bmm(x.transpose(1,2),torch.softmax(scores,dim=1).unsqueeze(2)).squeeze(2)
        return torch.tanh(self.fc2(torch.cat([context,last],dim=1)))

class Decoder(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.rnn=nn.LSTM(input_size=54,hidden_size=128,num_layers=4,bidirectional=True)
        self.attn_block=Attention()
        self.fc=nn.Linear(256,classes)
    def forward(self,x):
        output,_=self.rnn(x)
        return self.fc(self.attn_block(output))

class Model(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.decoder=Decoder(classes)
    def forward(self,poses):
        return self.decoder(poses.permute(0,2,1,3).flatten(start_dim=2))

# Model label order is the lexicographically sorted unique training Word column,
# not CSV row order or human numerical order (as the official dataset documents).
with open('tmp/models/include_metadata/Train_Test_Split/train_include.csv',encoding='utf-8-sig') as f:
    raw_classes=sorted({row['Word'] for row in csv.DictReader(f)})
with open('tmp/models/glosses.json',encoding='utf-8-sig') as f:
    mapping=dict(csv.reader(f)); mapping.pop('original',None)
import re
labels=[]
for label in raw_classes:
    stripped=re.sub(r'^\d+\.\s*','',label).strip()
    canonical=lambda text: re.sub(r'[^a-z0-9]','',re.sub(r'^\d+\.\s*','',text).casefold())
    match=next((value for key,value in mapping.items() if canonical(key)==canonical(stripped)),stripped)
    labels.append(match.replace('_',' ').upper())
clean={k.removeprefix('model.'):v for k,v in state.items()}
model=Model(len(labels))
model.load_state_dict(clean,strict=True)
model.eval()
torch.set_num_threads(1)
Path('public/models').mkdir(parents=True,exist_ok=True)
output='public/models/isl-words.onnx'
# Fixed 48 frames makes the browser interface and memory use predictable.
sample=torch.randn(1,2,48,27,generator=torch.Generator().manual_seed(42))
torch.onnx.export(model,sample,output,input_names=['poses'],output_names=['logits'],opset_version=17,dynamo=False)
meta={
    'classes':labels,'originalClasses':raw_classes,'shape':[1,2,48,27],
    'source':'https://github.com/AI4Bharat/OpenHands/releases/tag/checkpoints_v1',
    'checkpointSha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'onnxSha256':hashlib.sha256(Path(output).read_bytes()).hexdigest(),
    'normalization':'clip-level mean shoulder center and mean xy shoulder distance',
}
Path('public/models/isl-words.json').write_text(json.dumps(meta,indent=2))
with torch.no_grad():
    expected=model(sample).flatten().tolist()
Path('tests/fixtures').mkdir(parents=True,exist_ok=True)
Path('tests/fixtures/word-parity.json').write_text(json.dumps({'shape':[1,2,48,27],'input':sample.flatten().tolist(),'logits':expected}))
print('Exported',len(labels),'ISL word classes;',Path(output).stat().st_size,'bytes')

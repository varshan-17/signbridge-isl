import {mkdir,writeFile} from 'node:fs/promises';
await mkdir('tmp/model-upgrade',{recursive:true});
const sources={
  'include_stgcn.zip':'https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_stgcn.zip',
  'st_gcn.py':'https://raw.githubusercontent.com/AI4Bharat/OpenHands/main/openhands/models/encoder/graph/st_gcn.py',
  'transforms.py':'https://raw.githubusercontent.com/AI4Bharat/OpenHands/main/openhands/datasets/pose_transforms.py',
};
for (const [file,url] of Object.entries(sources)) {
  const response=await fetch(url,{signal:AbortSignal.timeout(90000)});
  if(!response.ok){console.log(file,response.status);continue;}
  await writeFile(`tmp/model-upgrade/${file}`,new Uint8Array(await response.arrayBuffer()));
  console.log('Saved',file);
}

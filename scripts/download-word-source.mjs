import {mkdir,writeFile} from 'node:fs/promises';
await mkdir('tmp/models',{recursive:true});
const sources={
 'include_lstm.zip':'https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_lstm.zip',
 'include_metadata.zip':'https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_metadata.zip',
 'glosses.json':'https://raw.githubusercontent.com/AI4Bharat/OpenHands/main/openhands/datasets/assets/include_metadata/normalized_glosses.csv',
};
for(const [name,url] of Object.entries(sources)) {
 const response=await fetch(url,{signal:AbortSignal.timeout(90000)});
 if(!response.ok) throw new Error(`Could not retrieve ${name}: ${response.status}`);
 await writeFile(`tmp/models/${name}`,new Uint8Array(await response.arrayBuffer()));
 console.log(`Saved ${name}`);
}

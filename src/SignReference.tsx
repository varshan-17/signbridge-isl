import {useEffect,useState} from 'react';
import {BookOpen,ExternalLink} from 'lucide-react';

const letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const dictionaryUrl='https://divyangjan.depwd.gov.in/islrtc/';
export function SignReference({mode='letter',initialLetter,wordLabel}:{mode?:'letter'|'word';initialLetter?:string;wordLabel?:string}) {
  const [letter,setLetter]=useState(initialLetter&&letters.includes(initialLetter)?initialLetter:'A');
  useEffect(()=>{if(initialLetter&&letters.includes(initialLetter))setLetter(initialLetter);},[initialLetter]);
  if(mode==='word') {
    const search=wordLabel?.trim();
    const href=search?`${dictionaryUrl}search.php?search=${encodeURIComponent(search)}`:dictionaryUrl;
    return <section className="sign-reference" aria-label="ISL word sign reference">
      <div className="reference-title"><h3><BookOpen size={16}/> ISL word sign reference</h3><span>Video guide</span></div>
      <div className="word-reference"><p>{search?<>Watch the dictionary entry for <strong>{search}</strong>. Word signs can use movement and facial expression, so a video is the clearest reference.</>:'Word signs can use movement and facial expression. Open the official dictionary to watch a sign before practising it.'}</p><a href={href} target="_blank" rel="noreferrer">{search?`Look up ${search}`:'Open ISL dictionary'} <ExternalLink size={13}/></a></div>
      <div className="reference-links"><a href="https://divyangjan.depwd.gov.in/islrtc/" target="_blank" rel="noreferrer">Indian Sign Language Dictionary · ISLRTC <ExternalLink size={11}/></a></div>
    </section>;
  }
  return <section className="sign-reference" aria-label="ISL sign image reference">
    <div className="reference-title"><h3><BookOpen size={16}/> ISL sign reference</h3><span>A–Z photographs</span></div>
    <div className="reference-content">
      <figure><img src={`/references/letters/${letter}.jpg`} alt={`Indian Sign Language fingerspelling for the letter ${letter}`} width="220" height="180"/><figcaption>Letter {letter}</figcaption></figure>
      <div className="reference-help"><label htmlFor="reference-letter">Choose a letter to practise</label><select id="reference-letter" value={letter} onChange={e=>setLetter(e.target.value)}>{letters.map(label=><option key={label}>{label}</option>)}</select><p>Follow the hand shape in the photo. Keep every signing hand in view and hold the letter steadily. Lower your hands briefly to repeat it.</p><p>These are fingerspelling references. Word signs can include movement; learn them from a video.</p></div>
    </div>
    <div className="reference-links"><a href="https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset" target="_blank" rel="noreferrer">Photos: RealSign · CC0 <ExternalLink size={11}/></a><a href="https://divyangjan.depwd.gov.in/islrtc/" target="_blank" rel="noreferrer">ISLRTC word dictionary <ExternalLink size={11}/></a></div>
  </section>;
}

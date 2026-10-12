import {useEffect,useState} from 'react';
import {BookOpen,ExternalLink} from 'lucide-react';

const letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const dictionaryUrl='https://divyangjan.depwd.gov.in/islrtc/';
export function SignReference({mode='letter',initialLetter,wordLabel}:{mode?:'letter'|'word'|'both';initialLetter?:string;wordLabel?:string}) {
  const [letter,setLetter]=useState(initialLetter&&letters.includes(initialLetter)?initialLetter:'A');
  useEffect(()=>{if(initialLetter&&letters.includes(initialLetter))setLetter(initialLetter);},[initialLetter]);
  const search=wordLabel?.trim();
  const wordGuide=<div className="word-reference"><p>{search?<>Find <strong>{search}</strong> in the official dictionary, watch its movement, then capture it with your upper body in view.</>:'Want to sign a whole word? Watch it in the official dictionary, then use Capture word sign in this same workspace.'}</p><a href={dictionaryUrl} target="_blank" rel="noreferrer">{search?`Find ${search} in the dictionary`:'Explore word signs'} <ExternalLink size={13}/></a></div>;
  if(mode==='word') {
    return <section className="sign-reference" aria-label="ISL word sign reference">
      <div className="reference-title"><h3><BookOpen size={16}/> ISL word sign reference</h3><span>Video guide</span></div>
      {wordGuide}
      <div className="reference-links"><a href="https://divyangjan.depwd.gov.in/islrtc/" target="_blank" rel="noreferrer">Indian Sign Language Dictionary · ISLRTC <ExternalLink size={11}/></a></div>
    </section>;
  }
  return <section className="sign-reference" aria-label="ISL sign image reference">
    <div className="reference-title"><h3><BookOpen size={16}/> Learn a sign</h3><span>{mode==='both'?'Letters + word videos':'A–Z photographs'}</span></div>
    <div className="reference-content">
      <figure><img src={`/references/letters/${letter}.jpg`} alt={`Indian Sign Language fingerspelling for the letter ${letter}`} width="220" height="180"/><figcaption>Letter {letter}</figcaption></figure>
      <div className="reference-help"><label htmlFor="reference-letter">Choose a letter to practise</label><select id="reference-letter" value={letter} onChange={e=>setLetter(e.target.value)}>{letters.map(label=><option key={label}>{label}</option>)}</select><p>Follow the hand shape in the photo. Keep every signing hand in view and hold the letter steadily. Lower your hands briefly to repeat it.</p><p>These are fingerspelling references. Word signs can include movement; learn them from a video.</p></div>
    </div>
    {mode==='both' && <div className="reference-word-guide"><h4>Word signs tell a bigger story</h4>{wordGuide}</div>}
    <div className="reference-links"><a href="https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset" target="_blank" rel="noreferrer">Photos: RealSign · CC0 <ExternalLink size={11}/></a><a href="https://divyangjan.depwd.gov.in/islrtc/" target="_blank" rel="noreferrer">ISLRTC word dictionary <ExternalLink size={11}/></a></div>
  </section>;
}

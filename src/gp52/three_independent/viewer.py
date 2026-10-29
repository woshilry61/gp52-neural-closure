"""Small standalone curve viewer; full metrics are computed before downsampling."""
import json,base64
from pathlib import Path
import numpy as np
from gp52.three.reference import ROOT
from .transfer import RUN
OUT=ROOT/'reports/independent-cfd-v1'
def main():
 d=dict(np.load(RUN/'report-curves.npz'));stride=4
 out={}
 def pack(name,z):
  z=np.ascontiguousarray(z,dtype='<f4');out[name]=dict(shape=z.shape,data=base64.b64encode(z.tobytes()).decode())
 pack('t',d['t'][::stride]);pack('truth',d['truth'][...,::stride])
 for mode in ['forward','inverse']:
  pack(f'c0_{mode}',np.stack([d[f'{b}-ordinary_1d-{mode}'][...,::stride] for b in ['cfd','sen']]))
  pack(f'nn_{mode}',np.stack([np.mean([d[f'{b}-neural_{s}-{mode}'][...,::stride] for s in [42,43,44]],axis=0) for b in ['cfd','sen']]))
 obs=np.load(RUN/'observations.npz');pack('obs',obs['observations'][32:48]);pack('ot',obs['sample_t'])
 html=r'''<!doctype html><meta charset="utf-8"><title>Independent three-vessel curves</title>
<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 18px;color:#153044;background:#f8fafb}h1{font-size:26px}select{padding:8px;margin:6px;border:1px solid #bdcbd1;border-radius:6px;background:white}#plots{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.panel{background:white;border:1px solid #dae3e8;border-radius:10px;padding:10px}canvas{width:100%;height:240px}p{line-height:1.5}.legend span{margin-right:22px}@media(max-width:900px){#plots{grid-template-columns:1fr}}</style>
<h1>Independent CFD blood — frozen model comparison</h1>
<p>Choose a held-out inlet case, blood input and inverse setting. The black reference and observations stay identical when switching blood inputs. Neural curves show the mean of the three frozen seeds; reported errors are averaged per seed. No training is performed.</p>
<div><label>Blood <select id="blood"><option value="0">Independent CFD</option><option value="1">Sen projected</option></select></label>
<label>Test case <select id="case"></select></label>
<label>Mode <select id="mode"><option value="forward">Known inlet</option><option value="inverse">Recovered inlet</option></select></label>
<label>Measurements <select id="layout"><option value="0">1 site</option><option value="1">2 sites</option><option value="2">3 sites</option></select></label>
<label>Noise <select id="rep"><option value="0">Clean</option><option value="1">Noise 1</option><option value="2">Noise 2</option><option value="3">Noise 3</option></select></label></div>
<p class="legend"><span style="color:black">● Independent reference</span><span style="color:#c76e24">● Fixed C0</span><span style="color:#167d9b">● Frozen neural</span><span style="color:#555">○ Observed Th/Fg</span></p>
<div id="plots"></div><p>Plot samples are reduced to every 8 ms for file size. All quantitative errors use the full 2 ms scoring grid. Inlet errors use 0–0.82 s; concentration errors use 0–1.23 s.</p>
<script>
const raw=PAYLOAD;
const data={};for(const [k,z] of Object.entries(raw)){const bytes=Uint8Array.from(atob(z.data),c=>c.charCodeAt(0));data[k]={shape:z.shape,v:new Float32Array(bytes.buffer)};}
function value(k,ix){let z=data[k],n=0;for(let i=0;i<ix.length;i++)n=n*z.shape[i]+ix[i];return z.v[n];}
function line(k,ix){const n=data[k].shape.at(-1);return Array.from({length:n},(_,i)=>value(k,[...ix,i]));}
const cas=document.querySelector('#case');for(let c=32;c<48;c++){let o=document.createElement('option');o.value=c-32;o.textContent=c;cas.append(o);}
const canvases=[];for(let s=0;s<3;s++)for(let v=0;v<3;v++){let panel=document.createElement('div');panel.className='panel';panel.innerHTML='<b>'+['Th','Fg','Fn'][s]+' — '+['Parent (30 mm)','Daughter 1 (10 mm)','Daughter 2 (17 mm)'][v]+'</b>';let c=document.createElement('canvas');panel.append(c);document.querySelector('#plots').append(panel);canvases.push([s,v,c]);}
function draw(){const b=+document.querySelector('#blood').value,c=+cas.value,mode=document.querySelector('#mode').value,r=+document.querySelector('#rep').value,l=+document.querySelector('#layout').value;
 document.querySelector('#layout').disabled=mode==='forward';document.querySelector('#rep').disabled=mode==='forward';
 const tt=Array.from(data.t.v);
 for(const [s,v,cv] of canvases){const tr=line('truth',[c,s,v]),ix=mode==='forward'?[b,c,s,v]:[b,c,r,l,s,v],a=line('c0_'+mode,ix),nn=line('nn_'+mode,ix);
 const points=mode==='inverse'&&s<2&&v<=l?line('obs',[c,r,s,v]):[];
 const vals=[...tr,...a,...nn,...points];let ymin=Math.min(...vals),ymax=Math.max(...vals),pad=Math.max(.02,.1*(ymax-ymin));ymin-=pad;ymax+=pad;
 const W=cv.clientWidth,H=240,dpr=devicePixelRatio||1;cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,W,H);
 const left=48,right=14,top=15,bottom=31;const x=t=>left+(W-left-right)*t/1.23,y=z=>H-bottom-(H-top-bottom)*(z-ymin)/(ymax-ymin);
 g.font='11px system-ui';for(let j=0;j<5;j++){let yy=ymin+(ymax-ymin)*j/4;g.strokeStyle='#edf0f2';g.beginPath();g.moveTo(left,y(yy));g.lineTo(W-right,y(yy));g.stroke();g.fillStyle='#586b78';g.fillText(yy.toFixed(2),3,y(yy)+4);}
 for(const t of [0,.41,.82,1.23]){g.fillStyle='#586b78';g.fillText(t.toFixed(2),x(t)-10,H-12);}
 for(const [z,color] of [[tr,'black'],[a,'#c76e24'],[nn,'#167d9b']]){g.strokeStyle=color;g.lineWidth=2;g.beginPath();z.forEach((q,i)=>i?g.lineTo(x(tt[i]),y(q)):g.moveTo(x(tt[i]),y(q)));g.stroke();}
 g.strokeStyle='#777';for(let i=0;i<points.length;i++){g.beginPath();g.arc(x(data.ot.v[i]),y(points[i]),3,0,2*Math.PI);g.stroke();}
 }}
document.querySelectorAll('select').forEach(z=>z.addEventListener('change',draw));window.addEventListener('resize',draw);draw();
</script>'''
 (OUT/'curves.html').write_text(html.replace('PAYLOAD',json.dumps(out)))
if __name__=='__main__':main()

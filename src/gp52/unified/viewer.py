"""Standalone all-case concentration viewer; metrics use the unsampled arrays."""
import json,base64
import numpy as np
from .protocol import ROOT,RUN,OUT,TEST,plan
def main():
 d=np.load(RUN/'report-curves.npz');payload={}
 def pack(k,z):
  z=np.ascontiguousarray(z,dtype='<f4');payload[k]=dict(shape=z.shape,data=base64.b64encode(z.tobytes()).decode())
 pack('t',d['t'][::4]);pack('inlet-truth',d['inlet-truth']);pack('knots',d['knots'])
 for name in ['single','three','seven']:
  pack(name+'-truth',d[name+'-truth'][...,::4])
  for method in ['c0','nn']:
   pack(name+'-'+method+'-inlet',d[name+'-'+method+'-inlet'])
   for mode in ['forward','inverse']:pack(name+'-'+method+'-'+mode,d[name+'-'+method+'-'+mode][...,::4])
  pack(name+'-obs',np.load(RUN/name/'observations.npz')['observations'][TEST])
 pack('ot',np.array(plan()['observation_times']))
 page=r'''<!doctype html><meta charset="utf-8"><title>Unified baseline concentration curves</title>
<style>body{font:16px system-ui;color:#173040;background:#f8fafb;max-width:1400px;margin:25px auto;padding:0 20px}p{line-height:1.6}select{padding:8px;margin:6px;border:1px solid #bccdd4;background:white;border-radius:6px}#plots{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.panel{background:white;padding:12px;border:1px solid #dae3e8;border-radius:8px}canvas{width:100%;height:230px}.legend span{margin-right:20px}@media(max-width:850px){#plots{grid-template-columns:1fr}}</style>
<p><a href="summary-zh.html">中文报告</a> · <a href="report.html">English report</a></p>
<h1>Unified independent-CFD baseline comparison</h1>
<p>All 32 test inlets. Concentrations use normalized model units. Neural curves use retrained seed 42; they are not an ensemble. Reported summary errors average the errors of all three training seeds. Black curves and observation values are identical for C0 and the neural method.</p>
<div><label>Network <select id="network"><option value="single">Single vessel</option><option value="three">Three vessels</option><option value="seven">Seven vessels</option></select></label>
<label>Case <select id="case"></select></label><label>Mode <select id="mode"><option value="forward">Known inlet</option><option value="inverse">Recovered inlet</option></select></label>
<label>Measurements <select id="layout"><option value="0">1 site</option><option value="1">2 sites</option><option value="2">3 sites</option></select></label>
<label>Noise <select id="rep"><option value="0">Clean</option><option value="1">Noise 1</option><option value="2">Noise 2</option><option value="3">Noise 3</option></select></label></div>
<p class="legend"><span style="color:black">● Independent reference</span><span style="color:#c77531">● Fixed C0</span><span style="color:#13868a">● Retrained neural baseline</span><span>○ Observed Th/Fg</span></p>
<div class="panel"><b>Th inlet — recovery scored over 0–0.82 s</b><canvas id="inlet"></canvas></div><br><div id="plots"></div><p>The charts display every fourth scoring sample. Full scores use 2 ms samples and all axial cells over 0–1.23 s. Inlet errors use 0–0.82 s. All three topologies use the same chemical inlet cases, partitions, noise schedule and scoring rules. Seven-vessel chart labels use one-based vessel numbers in model order; distances are measured within each vessel.</p>
<script>
const raw=PAYLOAD,data={};
for(const[k,z]of Object.entries(raw)){const b=Uint8Array.from(atob(z.data),c=>c.charCodeAt(0));data[k]={shape:z.shape,v:new Float32Array(b.buffer)};}
function value(k,ix){let z=data[k],n=0;for(let i=0;i<ix.length;i++)n=n*z.shape[i]+ix[i];return z.v[n];}
function line(k,ix){return Array.from({length:data[k].shape.at(-1)},(_,i)=>value(k,[...ix,i]));}
const cas=document.querySelector('#case');for(let i=80;i<112;i++){const o=document.createElement('option');o.value=i-80;o.textContent=i;cas.append(o);}
const panels=[];for(let s=0;s<3;s++)for(let p=0;p<3;p++){let div=document.createElement('div');div.className='panel';let title=document.createElement('b'),cv=document.createElement('canvas');div.append(title,cv);document.querySelector('#plots').append(div);panels.push([s,p,title,cv]);}
function drawInlet(name,c,mode,r,l){
const cv=document.querySelector('#inlet'),W=cv.clientWidth,H=230,dpr=devicePixelRatio||1;
cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);g.font='11px system-ui';
const xx=t=>48+(W-62)*t/1.23,yy=v=>H-30-(H-45)*(v+.05)/1.3;
for(const v of[0,.3,.6,.9,1.2]){g.strokeStyle='#e9eef1';g.beginPath();g.moveTo(48,yy(v));g.lineTo(W-14,yy(v));g.stroke();g.fillStyle='#526b78';g.fillText(v.toFixed(1),10,yy(v)+4);}
for(const t of[0,.41,.82,1.23])g.fillText(t.toFixed(2),xx(t)-10,H-11);
const series=[[line('inlet-truth',[c]),'black']];if(mode==='inverse'){series.push([line(name+'-c0-inlet',[c,r,l]),'#c77531'],[line(name+'-nn-inlet',[c,r,l]),'#13868a']);}
for(const[values,color]of series){const ts=[...data.knots.v,1.23],vs=[...values,values.at(-1)];g.strokeStyle=color;g.lineWidth=2;g.beginPath();vs.forEach((v,i)=>i?g.lineTo(xx(ts[i]),yy(v)):g.moveTo(xx(ts[i]),yy(v)));g.stroke();}
g.setLineDash([4,4]);g.strokeStyle='#bbc7cd';g.beginPath();g.moveTo(xx(.82),15);g.lineTo(xx(.82),H-30);g.stroke();
}
function draw(){
const name=document.querySelector('#network').value,c=+cas.value,mode=document.querySelector('#mode').value,r=+document.querySelector('#rep').value,l=+document.querySelector('#layout').value;
document.querySelector('#rep').disabled=mode==='forward';document.querySelector('#layout').disabled=mode==='forward';
drawInlet(name,c,mode,r,l);
const locations=name==='single'?['30 mm','6 mm','54 mm']:name==='three'?['Parent 30 mm','Daughter 1: 10 mm','Daughter 2: 17 mm']:['Parent 30 mm','Vessel 4: 19.5 mm','Vessel 7: 47 mm'];
for(const[s,p,title,cv]of panels){
title.textContent=['Th','Fg','Fn'][s]+' — '+locations[p];
const truth=line(name+'-truth',[c,s,p]),ix=mode==='forward'?[c,s,p]:[c,r,l,s,p];
const a=line(name+'-c0-'+mode,ix),b=line(name+'-nn-'+mode,ix),obs=mode==='inverse'&&s<2&&p<=l?line(name+'-obs',[c,r,s,p]):[];
let lo=Math.min(...truth,...a,...b,...obs),hi=Math.max(...truth,...a,...b,...obs),pad=Math.max(.02,(hi-lo)*.1);lo-=pad;hi+=pad;
const W=cv.clientWidth,H=230,scale=devicePixelRatio||1;cv.width=W*scale;cv.height=H*scale;const g=cv.getContext('2d');g.scale(scale,scale);
const x=t=>48+(W-62)*t/1.23,y=v=>H-30-(H-45)*(v-lo)/(hi-lo);g.font='11px system-ui';
for(let j=0;j<5;j++){const z=lo+(hi-lo)*j/4;g.strokeStyle='#e9eef1';g.beginPath();g.moveTo(48,y(z));g.lineTo(W-14,y(z));g.stroke();g.fillStyle='#526b78';g.fillText(z.toFixed(2),2,y(z)+4);}
for(const t of[0,.41,.82,1.23])g.fillText(t.toFixed(2),x(t)-10,H-11);
for(const[z,color]of[[truth,'black'],[a,'#c77531'],[b,'#13868a']]){g.strokeStyle=color;g.lineWidth=2;g.beginPath();z.forEach((v,i)=>i?g.lineTo(x(data.t.v[i]),y(v)):g.moveTo(x(data.t.v[i]),y(v)));g.stroke();}
g.strokeStyle='#666';obs.forEach((v,i)=>{g.beginPath();g.arc(x(data.ot.v[i]),y(v),3,0,Math.PI*2);g.stroke();});
}}
document.querySelectorAll('select').forEach(el=>el.addEventListener('change',draw));window.addEventListener('resize',draw);draw();
</script>'''
 (OUT/'curves.html').write_text(page.replace('PAYLOAD',json.dumps(payload)))
if __name__=='__main__':main()

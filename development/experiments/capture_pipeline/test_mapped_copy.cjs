'use strict';
const assert=require('node:assert/strict');
const {copyMapped,copyMappedTo}=require('./d3d_bridge.cjs');
const koffi=require('../../local/electron_worker/node_modules/koffi');
const crt=koffi.load('msvcrt.dll');
const copy=crt.func('void *memcpy(void *,const void *,size_t)');
for(const [row,height,pad] of [[12,7,0],[12,7,4],[7680,945,0],[7680,945,256]]) {
  const pitch=row+pad,source=Buffer.alloc(pitch*height);
  for(let i=0;i<source.length;i++)source[i]=(i*17+(i>>8))%256;
  const ptr=koffi.address(source),a=Buffer.alloc(row*height),b=Buffer.alloc(row*height);
  assert.equal(copyMapped(copy,a,ptr,pitch,row,height,false),height);
  assert.equal(copyMapped(copy,b,ptr,pitch,row,height,true),pad?height:1);
  assert.deepEqual(a,b);
  const direct=Buffer.alloc(row*height);
  assert.equal(copyMappedTo(copy,{pointer:koffi.address(direct),bytes:direct.length},ptr,pitch,row,height),pad?height:1);
  assert.deepEqual(a,direct);
  for(let y=0;y<height;y++)assert.deepEqual(b.subarray(y*row,(y+1)*row),source.subarray(y*pitch,y*pitch+row));
}
for(const args of [[0n,12,12,1],[1n,8,12,1],[1n,12,12,0],[1n,12,12,2]])
  assert.throws(()=>copyMapped(copy,Buffer.alloc(12),...args),/Invalid/);
assert.throws(()=>copyMappedTo(copy,{pointer:0n,bytes:12},1n,12,12,1),/Invalid/);
assert.throws(()=>copyMappedTo(copy,{pointer:1n,bytes:8},1n,12,12,1),/Invalid/);
// Measure the exact installed native FFI path, separate from GPU/capture latency.
const row=7680,height=945,source=Buffer.alloc(row*height,42),ptr=koffi.address(source),dest=Buffer.alloc(source.length);
const timings={};
for(const bulk of [false,true]) {
  for(let i=0;i<20;i++)copyMapped(copy,dest,ptr,row,row,height,bulk);
  const values=[];
  for(let i=0;i<150;i++){const t=performance.now();copyMapped(copy,dest,ptr,row,row,height,bulk);values.push(performance.now()-t);}
  values.sort((a,b)=>a-b);timings[bulk?'bulk':'rows']={median_ms:values[75],p95_ms:values[142]};
}
console.log(JSON.stringify({status:'PASS',native_pixel_parity:true,cases:8,timings},null,2));

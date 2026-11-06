'use strict';
const assert=require('node:assert/strict');
const koffi=require('../../local/electron_worker/node_modules/koffi');
const create=require('../../../tools/electron_worker/shared_frame.cjs');
const {copyMappedTo}=require('./d3d_bridge.cjs');
const kernel=koffi.load('kernel32.dll'),crt=koffi.load('msvcrt.dll');
const open=kernel.func('void* __stdcall OpenFileMappingW(uint32_t,int,str16)');
const map=kernel.func('void* __stdcall MapViewOfFile(void*,uint32_t,uint32_t,uint32_t,size_t)');
const unmap=kernel.func('int __stdcall UnmapViewOfFile(void*)');
const close=kernel.func('int __stdcall CloseHandle(void*)');
const copy=crt.func('void *memcpy(void *,const void *,size_t)');
const width=64,height=48,bytes=width*height*4;
const slot=create('Local\\RSReadbackTest_'+process.pid+'_'+Date.now(),bytes);
const handle=open(4,0,slot.name),view=map(handle,4,0,0,bytes+64);
assert.ok(handle&&view);
const read=(offset,len)=>{const b=Buffer.alloc(len);copy(b,view+BigInt(offset),len);return b;};
try {
  const source=Buffer.alloc(bytes);for(let i=0;i<bytes;i++)source[i]=(i*23+i%7)%256;
  slot.publish(source,1,width,height);
  const expected=read(64,bytes);
  slot.publishFrom(target=>{
    assert.equal(read(0,4).readUInt32LE(),0,'slot must be invalid before writing pixels');
    copyMappedTo(copy,target,koffi.address(source),width*4,width*4,height);
    assert.equal(read(0,4).readUInt32LE(),0,'writer must not publish early');
    return {byte_length:bytes};
  },2,width,height);
  assert.deepEqual(read(64,bytes),expected);
  const header=read(0,32);
  assert.equal(header.readUInt32LE(),0x52534631);assert.equal(header.readBigUInt64LE(16),2n);
  assert.throws(()=>slot.publishFrom(()=>{throw Error('simulated GPU fault');},3,width,height),/simulated/);
  assert.equal(read(0,4).readUInt32LE(),0);
  assert.throws(()=>slot.publishFrom(()=>({byte_length:bytes-1}),4,width,height),/Incomplete/);
  assert.equal(read(0,4).readUInt32LE(),0);
  slot.publish(source,5,width,height);
  assert.equal(read(16,8).readBigUInt64LE(),5n);assert.deepEqual(read(64,bytes),expected);
  assert.throws(()=>slot.publishFrom(()=>null,6,width+1,height),/Invalid/);
  slot.close();assert.throws(()=>slot.publishFrom(()=>null,6,width,height),/closed/);
  console.log(JSON.stringify({status:'PASS',pixel_parity:true,publication_order:true,failed_write_unpublished:true,close_checked:true}));
} finally {unmap(view);close(handle);slot.close();}

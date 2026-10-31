'use strict';
// One slot per renderer. Publishing socket metadata transfers ownership to
// Python; no overwrite is possible until Python requests the next frame.
module.exports=function createSharedFrame(name,bytes) {
  const path=require('node:path');
  const koffi=require(path.resolve(__dirname,'../../development/local/electron_worker/node_modules/koffi'));
  const kernel=koffi.load('kernel32.dll');
  const crt=koffi.load('msvcrt.dll');
  const create=kernel.func('void* __stdcall CreateFileMappingW(intptr_t, void*, uint32_t, uint32_t, uint32_t, str16)');
  const map=kernel.func('void* __stdcall MapViewOfFile(void*, uint32_t, uint32_t, uint32_t, size_t)');
  const unmap=kernel.func('int __stdcall UnmapViewOfFile(void*)');
  const close=kernel.func('int __stdcall CloseHandle(void*)');
  const copy=crt.func('void* memcpy(void*, const void*, size_t)');
  const handle=create(-1,null,4,0,bytes+64,name);
  if(!handle)throw Error('CreateFileMappingW failed');
  const view=map(handle,2,0,0,bytes+64);
  if(!view){close(handle);throw Error('MapViewOfFile failed');}
  let closed=false;
  return {name,
    publish(pixels,serial,width,height) {
      if(closed)throw Error('Shared mapping is closed');
      if(pixels.length!==bytes)throw Error('Shared frame byte count changed');
      const header=Buffer.alloc(32);
      header.writeUInt32LE(0x52534631,0);header.writeUInt32LE(1,4);
      header.writeUInt32LE(width,8);header.writeUInt32LE(height,12);
      header.writeBigUInt64LE(BigInt(serial),16);header.writeBigUInt64LE(BigInt(bytes),24);
      // Invalidate publication before touching pixels, including protection
      // against an accidental concurrent request/reader.
      copy(view,Buffer.alloc(4),4);
      copy(view+64n,pixels,bytes);
      copy(view,header,header.length);
    },
    close(){if(!closed){closed=true;unmap(view);close(handle);}}
  };
};

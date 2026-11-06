'use strict';
// Windows x64 D3D11 bridge using the already installed Koffi native FFI.
// No browser rebuild, downloaded addon, or compilation toolchain required.
// COM slots follow Microsoft's d3d11.h/d3d11_1.h. All GPU work completes
// before the caller releases Electron's shared texture lease.
const path=require('node:path');
const koffi=require(path.resolve(__dirname,'../../local/electron_worker/node_modules/koffi'));
const signatures=new Map();
function copyMapped(copy,pixels,ptr,pitch,row,height,bulk=true) {
  if(!ptr||!Number.isSafeInteger(row)||row<=0||!Number.isSafeInteger(height)||height<=0||
     !Number.isSafeInteger(pitch)||pitch<row||pixels.length!==row*height)
    throw Error('Invalid mapped texture copy layout');
  // Padding is not image data. Only collapse copies when row pitch is exact.
  if(bulk&&pitch===row){copy(pixels,ptr,pixels.length);return 1;}
  for(let y=0;y<height;y++)copy(pixels.subarray(y*row,(y+1)*row),ptr+BigInt(y*pitch),row);
  return height;
}
function copyMappedTo(copy,target,ptr,pitch,row,height) {
  if(!target||typeof target.pointer!=='bigint'||target.pointer===0n||!ptr||
     !Number.isSafeInteger(row)||row<=0||!Number.isSafeInteger(height)||height<=0||
     !Number.isSafeInteger(pitch)||pitch<row||target.bytes!==row*height)
    throw Error('Invalid mapped texture destination');
  if(pitch===row){copy(target.pointer,ptr,target.bytes);return 1;}
  for(let y=0;y<height;y++)copy(target.pointer+BigInt(y*row),ptr+BigInt(y*pitch),row);
  return height;
}
function method(object,slot,result,params,...args) {
  if(!object)throw Error('Null COM object');
  const key=result+'('+params.join(',')+')';
  if(!signatures.has(key))signatures.set(key,koffi.proto('__stdcall','fn'+signatures.size,result,['void *',...params]));
  const table=koffi.decode(object,'void *');
  const address=koffi.decode(table+BigInt(slot*8),'void *');
  return koffi.call(address,signatures.get(key),object,...args);
}
function check(hr,name) {if(hr<0)throw Error(name+' HRESULT 0x'+(hr>>>0).toString(16));}
function release(p) {if(p)method(p,2,'uint32_t',[]);}
function guid(text) {
  const hex=text.replaceAll('-','');const b=Buffer.alloc(16);
  b.writeUInt32LE(parseInt(hex.slice(0,8),16));b.writeUInt16LE(parseInt(hex.slice(8,12),16),4);
  b.writeUInt16LE(parseInt(hex.slice(12,16),16),6);Buffer.from(hex.slice(16),'hex').copy(b,8);return b;
}
const IID_DEVICE1=guid('a04bfb29-08ef-43d6-a49c-a9bdbdcbe686');
const IID_TEXTURE=guid('6f15aaf2-d208-4e89-9ab4-489535d34f9c');
function query(p,iid){const out=[null];check(method(p,0,'int32_t',['void *','_Out_ void **'],iid,out),'QueryInterface');return out[0];}
function descriptor(w,h,format,usage,bind,cpu){
  const b=Buffer.alloc(44);[w,h,1,1,format,1,0,usage,bind,cpu,0].forEach((v,i)=>b.writeUInt32LE(v,i*4));return b;
}
const shader=`Texture2D<float4> inputImage:register(t0);
RWTexture2D<float4> outputImage:register(u0);
[numthreads(8,8,1)] void main(uint3 id:SV_DispatchThreadID) {
 uint w,h,sw,sh; outputImage.GetDimensions(w,h);inputImage.GetDimensions(sw,sh);
 if(id.x>=w||id.y>=h)return;
 float2 p=(float2(id.xy)+0.5)*float2(sw,sh)/float2(w,h)-0.5;
 int2 lo=int2(floor(p));float2 a=frac(p);int2 hi=int2(sw-1,sh-1);
 float4 c00=inputImage.Load(int3(clamp(lo,int2(0,0),hi),0));
 float4 c10=inputImage.Load(int3(clamp(lo+int2(1,0),int2(0,0),hi),0));
 float4 c01=inputImage.Load(int3(clamp(lo+int2(0,1),int2(0,0),hi),0));
 float4 c11=inputImage.Load(int3(clamp(lo+int2(1,1),int2(0,0),hi),0));
 // Output resource is RGBA8; write BGRA bytes for the existing consumer.
 outputImage[id.xy]=lerp(lerp(c00,c10,a.x),lerp(c01,c11,a.x),a.y).bgra;
}`;
class Bridge {
  constructor(width,height) {
    if(process.platform!=='win32'||process.arch!=='x64')throw Error('Windows x64 required');
    this.width=width;this.height=height;this.owned=[];
    try {
      const d3d=koffi.load('d3d11.dll');
      // Koffi libraries must outlive COM objects whose vtables point into them.
      this.libraries=[d3d];
      const create=d3d.func('int32_t __stdcall D3D11CreateDevice(void *, uint32_t, void *, uint32_t, void *, uint32_t, uint32_t, _Out_ void **, _Out_ uint32_t *, _Out_ void **)');
      const dev=[null],ctx=[null],level=[0];
      check(create(null,1,null,0x20,null,0,7,dev,level,ctx),'D3D11CreateDevice');
      this.device=dev[0];this.context=ctx[0];this.owned.push(this.device,this.context);
      this.device1=query(this.device,IID_DEVICE1);this.owned.push(this.device1);
      const compiler=koffi.load('d3dcompiler_47.dll');
      this.libraries.push(compiler);
      const compile=compiler.func('int32_t __stdcall D3DCompile(const void *, size_t, str, void *, void *, str, str, uint32_t, uint32_t, _Out_ void **, _Out_ void **)');
      const code=[null],errors=[null];const bytes=Buffer.from(shader);
      const hr=compile(bytes,bytes.length,null,null,null,'main','cs_5_0',1<<15,0,code,errors);
      release(errors[0]);check(hr,'D3DCompile');
      try {
        const ptr=method(code[0],3,'void *',[]),len=method(code[0],4,'size_t',[]),out=[null];
        check(method(this.device,18,'int32_t',['void *','size_t','void *','_Out_ void **'],ptr,len,null,out),'CreateComputeShader');
        this.shader=out[0];this.owned.push(this.shader);
      } finally {release(code[0]);}
      this.output=this.texture(descriptor(width,height,28,0,0x80,0));
      this.staging=this.texture(descriptor(width,height,28,3,0,0x20000));
      const uav=[null];check(method(this.device,8,'int32_t',['void *','void *','_Out_ void **'],this.output,null,uav),'CreateUAV');
      this.uav=uav[0];this.owned.push(this.uav);
      const crt=koffi.load('msvcrt.dll');this.libraries.push(crt);
      this.copy=crt.func('void *memcpy(void *,const void *,size_t)');
    }catch(e){this.close();throw e;}
  }
  texture(desc){const out=[null];check(method(this.device,5,'int32_t',['void *','void *','_Out_ void **'],desc,null,out),'CreateTexture2D');this.owned.push(out[0]);return out[0];}
  read(info,sourceWidth,sourceHeight,target=null) {
    const trace=message=>{if(process.env.RS_GPU_TRACE==='1')require('node:fs').writeSync(2,message+'\n');};
    if(!['rgba','bgra'].includes(info.pixelFormat))throw Error('Unsupported shared format '+info.pixelFormat);
    const handle=info.handle.ntHandle;
    if(!Buffer.isBuffer(handle)||handle.length!==8)throw Error('Invalid x64 shared NT handle');
    if(info.codedSize.width!==sourceWidth||info.codedSize.height!==sourceHeight||
       info.visibleRect.x!==0||info.visibleRect.y!==0||info.visibleRect.width!==sourceWidth||info.visibleRect.height!==sourceHeight)
      throw Error('Shared texture geometry changed');
    let texture=null,view=null,mapped=false;
    const start=performance.now();
    try {
      trace('open texture');
      const tex=[null];check(method(this.device1,48,'int32_t',['void *','void *','_Out_ void **'],handle.readBigUInt64LE(),IID_TEXTURE,tex),'OpenSharedResource1');texture=tex[0];
      trace('descriptor');
      const desc=Buffer.alloc(44);method(texture,10,'void',['void *'],desc);
      const expected=info.pixelFormat==='bgra'?87:28;
      if(desc.readUInt32LE(0)!==sourceWidth||desc.readUInt32LE(4)!==sourceHeight||desc.readUInt32LE(16)!==expected||desc.readUInt32LE(20)!==1)
        throw Error('D3D texture descriptor mismatch');
      trace('view');
      const srv=[null];check(method(this.device,7,'int32_t',['void *','void *','_Out_ void **'],texture,null,srv),'CreateSRV');view=srv[0];
      trace('bind');
      method(this.context,67,'void',['uint32_t','uint32_t','void **'],0,1,[view]);
      method(this.context,68,'void',['uint32_t','uint32_t','void **','void *'],0,1,[this.uav],null);
      method(this.context,69,'void',['void *','void *','uint32_t'],this.shader,null,0);
      trace('dispatch');
      method(this.context,41,'void',['uint32_t','uint32_t','uint32_t'],Math.ceil(this.width/8),Math.ceil(this.height/8),1);
      method(this.context,67,'void',['uint32_t','uint32_t','void **'],0,1,[null]);
      method(this.context,68,'void',['uint32_t','uint32_t','void **','void *'],0,1,[null],null);
      trace('copy');
      method(this.context,47,'void',['void *','void *'],this.staging,this.output);
      const beforeMap=performance.now(),map=Buffer.alloc(16);
      // Blocking Map completes the GPU copy, keeping Electron's lease alive.
      trace('map');
      check(method(this.context,14,'int32_t',['void *','uint32_t','uint32_t','uint32_t','void *'],this.staging,0,1,0,map),'Map');mapped=true;
      const afterMap=performance.now(),bytes=this.width*this.height*4,pixels=target?null:Buffer.allocUnsafe(bytes);
      const ptr=map.readBigUInt64LE(0),pitch=map.readUInt32LE(8),row=this.width*4;
      if(!ptr||pitch<row)throw Error('Invalid mapped texture pitch');
      trace('memcpy');
      const copyCalls=target?copyMappedTo(this.copy,target,ptr,pitch,row,this.height):
        copyMapped(this.copy,pixels,ptr,pitch,row,this.height,process.env.RS_GPU_BULK_COPY!=='0');
      return {pixels,byte_length:bytes,profile:{gpu_submit_ms:beforeMap-start,gpu_map_wait_ms:afterMap-beforeMap,
        mapped_copy_ms:performance.now()-afterMap,gpu_bridge_ms:performance.now()-start,
        mapped_row_pitch:pitch,mapped_copy_calls:copyCalls,direct_shared_readback:!!target}};
    } finally {
      if(mapped)method(this.context,15,'void',['void *','uint32_t'],this.staging,0);
      release(view);release(texture);
    }
  }
  close(){if(this.context)method(this.context,110,'void',[]);for(const p of this.owned.reverse())release(p);this.owned=[];this.context=null;}
}
module.exports={Bridge,guid,copyMapped,copyMappedTo};

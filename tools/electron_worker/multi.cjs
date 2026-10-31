'use strict';
// Optional shared Electron host; each game retains its own renderer/session/slot.
const {app,BrowserWindow}=require('electron');
const net=require('node:net'),fs=require('node:fs'),path=require('node:path');
const args=Object.fromEntries(process.argv.slice(2).filter(v=>v.startsWith('--')&&v.includes('='))
 .map(v=>{const i=v.indexOf('=');return [v.slice(2,i),v.slice(i+1)];}));
const width=Number(args.width),height=Number(args.height),count=Number(args.count);
if(!args.profile||!args.ready||!args.token||!args['cdp-port']||!Number.isInteger(count)||count<1||count>6||
 !Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1)throw Error('Invalid renderer group');
app.setPath('userData',args.profile);
app.setPath('crashDumps',path.join(args.profile,'crashes'));
app.commandLine.appendSwitch('remote-debugging-port',args['cdp-port']);
app.commandLine.appendSwitch('remote-debugging-address','127.0.0.1');
app.commandLine.appendSwitch('force-device-scale-factor','1');
for(const flag of ['disable-background-timer-throttling','disable-backgrounding-occluded-windows','disable-renderer-backgrounding'])app.commandLine.appendSwitch(flag);
let quitting=false;
const cleanup=[];
function reply(socket,metadata,pixels){const header=Buffer.from(JSON.stringify(metadata)),size=Buffer.alloc(4);
 size.writeUInt32LE(header.length);socket.cork();socket.write(size);socket.write(header);if(pixels)socket.write(pixels);socket.uncork();}
function cancel(socket){if(socket.frameRequest){clearTimeout(socket.frameRequest.timer);clearTimeout(socket.frameRequest.invalidate);}socket.frameRequest=null;}
async function create(index){
 const initial='about:blank#rs_worker_'+index,token=args.token+'_'+index;
 const win=new BrowserWindow({width,height,useContentSize:true,show:false,frame:false,skipTaskbar:true,
  webPreferences:{partition:'persist:rs_worker_'+index,offscreen:true,backgroundThrottling:false,nodeIntegration:false,contextIsolation:true,sandbox:true}});
 win.webContents.setFrameRate(60);win.webContents.setAudioMuted(true);
 win.webContents.setWindowOpenHandler(()=>({action:'deny'}));
 const allowed=url=>{if(url===initial||url==='about:blank')return true;try{const u=new URL(url);
  return u.protocol==='http:'&&['localhost','127.0.0.1'].includes(u.hostname)&&u.port==='8000';}catch{return false;}};
 win.webContents.on('will-navigate',(event,url)=>{if(!allowed(url))event.preventDefault();});
 win.webContents.on('will-frame-navigate',event=>{if(!allowed(event.url))event.preventDefault();});
 const shared=args.transport==='shared_memory'?require('./shared_frame.cjs')('Local\\RSFrame_'+require('node:crypto').randomBytes(24).toString('hex'),width*height*4):null;
 const clients=new Set();let serial=0,closed=false;
 const server=net.createServer(socket=>{
  if(clients.size||closed){socket.destroy();return;}clients.add(socket);socket.setNoDelay(true);let input='';
  socket.on('data',chunk=>{input+=chunk.toString('utf8');if(input.length>4096){socket.destroy();return;}
   let newline;while((newline=input.indexOf('\n'))>=0){let request;const line=input.slice(0,newline);input=input.slice(newline+1);
    try{request=JSON.parse(line);}catch{socket.destroy();return;}if(request.token!==token){socket.destroy();return;}
    socket.authenticated=true;
    if(request.op==='close'){cancel(socket);socket.end();win.close();return;}
    if(request.op!=='frame'||socket.frameRequest){socket.destroy();return;}
    socket.frameRequest={after:serial};socket.frameRequest.invalidate=setTimeout(()=>{if(!closed)win.webContents.invalidate();},80);
    socket.frameRequest.timer=setTimeout(()=>{cancel(socket);reply(socket,{error:'No fresh offscreen paint within 500ms'});},500);
   }});
  socket.on('error',()=>{});socket.on('close',()=>{cancel(socket);clients.delete(socket);if(socket.authenticated&&!closed&&!quitting)win.close();});
 });
 win.webContents.on('paint',(_event,dirty,image)=>{
  serial++;const paint_wall=Date.now()/1000;
  for(const socket of clients){if(!socket.frameRequest||serial<=socket.frameRequest.after)continue;
   const dimensions=image.getSize();if(dimensions.width!==width||dimensions.height!==height){cancel(socket);reply(socket,{error:'Offscreen frame geometry changed'});continue;}
   const pixels=image.toBitmap();cancel(socket);if(pixels.length!==width*height*4){reply(socket,{error:'Unexpected bitmap length'});continue;}
   if(shared)shared.publish(pixels,serial,width,height);
   reply(socket,{width,height,channels:4,format:'BGRA',serial,paint_wall,byte_length:pixels.length,dirty,
    transport:shared?'shared_memory':'socket',clock_kind:'Electron paint callback, not GPU presentation'},shared?null:pixels);
  }});
 const close=()=>{if(closed)return;closed=true;for(const s of clients){cancel(s);s.destroy();}if(shared)shared.close();server.close();};
 cleanup.push(close);win.on('closed',close);
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));await win.loadURL(initial);
 return {port:server.address().port,debug_port:Number(args['cdp-port']),initial_url:initial,token,
  electron:process.versions.electron,chrome:process.versions.chrome,width,height,visible:false,offscreen:true,audio_muted:true,
  transport:shared?'shared_memory':'socket',shared_name:shared?shared.name:null,shared_host:true,worker:index};
}
app.whenReady().then(async()=>{const windows=[];for(let i=0;i<count;i++)windows.push(await create(i));
 fs.writeFileSync(args.ready+'.partial',JSON.stringify({windows}));fs.renameSync(args.ready+'.partial',args.ready);
}).catch(error=>{console.error(error);app.exit(1);});
app.on('window-all-closed',()=>app.quit());
app.on('before-quit',()=>{quitting=true;for(const close of cleanup)close();});
process.on('uncaughtException',error=>{console.error(error);app.exit(1);});

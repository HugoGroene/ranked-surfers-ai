// Actual on-screen Chromium game. No OSR and no screenshot encoding.
'use strict';
const {app, BrowserWindow, screen} = require('electron');
const fs = require('node:fs'), net = require('node:net'), path = require('node:path');
const args = Object.fromEntries(process.argv.slice(2).filter(x=>x.startsWith('--')&&x.includes('='))
  .map(x=>{const i=x.indexOf('=');return [x.slice(2,i),x.slice(i+1)];}));
const width=Number(args.width), height=Number(args.height);
const localPort=args['local-port'] || '8000';
if(!['8000','8010','8012'].includes(localPort))throw Error('Unsupported local port');
if(width!==1920 || height!==945 || !args.profile || !args.ready || !args.token || !args['cdp-port'])
  throw Error('Invalid visible evaluation configuration');
app.setPath('userData',args.profile);
app.commandLine.appendSwitch('remote-debugging-port',args['cdp-port']);
app.commandLine.appendSwitch('remote-debugging-address','127.0.0.1');
app.commandLine.appendSwitch('force-device-scale-factor','1');
for(const flag of ['enable-gpu','disable-background-timer-throttling',
  'disable-backgrounding-occluded-windows','disable-renderer-backgrounding']) app.commandLine.appendSwitch(flag);
const shared=require('../../../tools/electron_worker/shared_frame.cjs')(
  'Local\\RSFrame_'+require('node:crypto').randomBytes(24).toString('hex'),width*height*4);
let win, server, serial=0, quitting=false;
const clients=new Set(), now=()=>process.hrtime.bigint();
const ms=(a,b)=>Number(a-b)/1e6;
function reply(socket,data){
  if(socket.destroyed)return;
  const bytes=Buffer.from(JSON.stringify(data)), header=Buffer.alloc(4);
  header.writeUInt32LE(bytes.length);socket.cork();socket.write(header);socket.write(bytes);socket.uncork();
}
function allowed(url){
  if(url==='about:blank')return true;
  try {const u=new URL(url);return u.protocol==='http:'&&['localhost','127.0.0.1'].includes(u.hostname)&&u.port===localPort;}
  catch{return false;}
}
async function capture(socket){
  if(socket.busy){socket.destroy();return;}
  socket.busy=true;
  const began=now();
  try {
    if(!win.isVisible() || win.isMinimized())throw Error('Keep the live evaluation window visible');
    // Fresh capture request, no cached application frame queue. Native readback
    // timestamp is not a verified Unity tick or GPU presentation timestamp.
    const image=await win.webContents.capturePage({x:0,y:0,width,height});
    if(quitting || socket.destroyed)return;
    const read=now(), dimensions=image.getSize();
    if(dimensions.width!==width||dimensions.height!==height)throw Error('Visible capture size changed');
    const pixels=image.toBitmap(), extracted=now();
    if(pixels.length!==width*height*4)throw Error('Visible bitmap format changed');
    serial++;shared.publish(pixels,serial,width,height);
    reply(socket,{width,height,channels:4,format:'BGRA',serial,byte_length:pixels.length,
      transport:'shared_memory',shared_copy_ms:ms(now(),extracted),
      profile:{native_capture_ms:ms(read,began),bitmap_extract_ms:ms(extracted,read),
               node_request_to_send_ms:ms(now(),began)},
      clock_kind:'capturePage request/readback; GPU presentation time unverified'});
  }catch(error){reply(socket,{error:String(error)});}
  finally{socket.busy=false;}
}
app.whenReady().then(async()=>{
  const area=screen.getPrimaryDisplay().workArea;
  if(area.width<width||area.height<height)throw Error('Display work area cannot fit the 1920x945 game');
  win=new BrowserWindow({width,height,x:area.x,y:area.y+Math.floor((area.height-height)/2),
    useContentSize:true,show:true,frame:false,resizable:false,maximizable:false,fullscreenable:false,
    title:'Live PPO evaluation - Ctrl+C in terminal to stop',skipTaskbar:false,
    webPreferences:{offscreen:false,backgroundThrottling:false,nodeIntegration:false,contextIsolation:true,sandbox:true}});
  win.webContents.setAudioMuted(true);
  win.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  win.webContents.on('will-navigate',(event,url)=>{if(!allowed(url))event.preventDefault();});
  win.webContents.on('will-frame-navigate',event=>{if(!allowed(event.url))event.preventDefault();});
  server=net.createServer(socket=>{
    if(clients.size){socket.destroy();return;}
    clients.add(socket);socket.setNoDelay(true);let input='';
    socket.on('data',chunk=>{
      input+=chunk.toString('utf8');if(input.length>4096){socket.destroy();return;}
      let end;
      while((end=input.indexOf('\n'))>=0){
        let request;try{request=JSON.parse(input.slice(0,end));}catch{socket.destroy();return;}
        input=input.slice(end+1);
        if(request.token!==args.token){socket.destroy();return;}
        socket.authenticated=true;
        if(request.op==='close'){app.quit();return;}
        if(request.op!=='frame'){socket.destroy();return;}
        void capture(socket);
      }
    });
    socket.on('error',()=>{});
    socket.on('close',()=>{clients.delete(socket);if(socket.authenticated&&!quitting)app.quit();});
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  await win.loadURL('about:blank');win.show();win.focus();
  fs.writeFileSync(args.ready+'.partial',JSON.stringify({port:server.address().port,
    hwnd:win.getNativeWindowHandle().readBigUInt64LE(0).toString(),
    width,height,visible:win.isVisible(),offscreen:win.webContents.isOffscreen(),
    transport:'shared_memory',shared_name:shared.name,backend:'visible_native_bitmap'}));
  fs.renameSync(args.ready+'.partial',args.ready);
}).catch(error=>{console.error(error.stack);app.exit(1);});
app.on('window-all-closed',()=>app.quit());
app.on('before-quit',()=>{quitting=true;for(const client of clients)client.destroy();
  if(server)server.close();shared.close();});
process.on('uncaughtException',error=>{console.error(error.stack);app.exit(1);});

// Isolated, invisible LOCAL Practice renderer. One demanded raw frame at a time.
'use strict';
const {app, BrowserWindow} = require('electron');
process.on('uncaughtException',error=>{console.error(error.stack);app.exit(1);});
const net = require('node:net');
const fs = require('node:fs');
const path = require('node:path');
const args = Object.fromEntries(process.argv.slice(2).filter(x=>x.startsWith('--') && x.includes('='))
  .map(x=>{const i=x.indexOf('=');return [x.slice(2,i),x.slice(i+1)];}));
const width=Number(args.width), height=Number(args.height);
const profileTimings=args['profile-timings']==='1';
const nowNs=()=>process.hrtime.bigint();
const elapsedMs=(end,start)=>Number(end-start)/1e6;
if (!args.profile || !args.ready || !args.token || !args['cdp-port'] ||
    !Number.isInteger(width) || !Number.isInteger(height) || width<1 || height<1) {
  throw Error('Missing isolated renderer configuration');
}
app.setPath('userData',args.profile);
app.setPath('crashDumps',path.join(args.profile,'crashes'));
app.commandLine.appendSwitch('remote-debugging-port',args['cdp-port']);
app.commandLine.appendSwitch('remote-debugging-address','127.0.0.1');
app.commandLine.appendSwitch('force-device-scale-factor','1');
for (const flag of ['disable-background-timer-throttling','disable-backgrounding-occluded-windows',
                    'disable-renderer-backgrounding']) app.commandLine.appendSwitch(flag);
let win, server, serial=0, quitting=false;
const shared=args.transport==='shared_memory'
  ?require('./shared_frame.cjs')('Local\\RSFrame_'+require('node:crypto').randomBytes(24).toString('hex'),width*height*4):null;
const clients=new Set();
function reply(socket,metadata,pixels) {
  const header=Buffer.from(JSON.stringify(metadata));
  const size=Buffer.alloc(4);size.writeUInt32LE(header.length);
  socket.cork();socket.write(size);socket.write(header);
  if (pixels) socket.write(pixels);
  socket.uncork();
}
function cancel(socket) {
  if (socket.frameRequest) { clearTimeout(socket.frameRequest.timer);clearTimeout(socket.frameRequest.invalidate); }
  socket.frameRequest=null;
}
app.whenReady().then(async()=>{
  win=new BrowserWindow({width,height,useContentSize:true,show:false,frame:false,skipTaskbar:true,
    webPreferences:{offscreen:true,backgroundThrottling:false,nodeIntegration:false,
      contextIsolation:true,sandbox:true}});
  win.webContents.setFrameRate(60);
  win.webContents.setAudioMuted(true);
  win.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  // Block navigation away from the local Practice copy (allow blank test setup).
  function allowed(url) {
    if(url==='about:blank') return true;
    try {const u=new URL(url);return u.protocol==='http:' &&
      ['localhost','127.0.0.1'].includes(u.hostname) && u.port==='8000';} catch {return false;}
  }
  win.webContents.on('will-navigate',(event,url)=>{if(!allowed(url))event.preventDefault();});
  win.webContents.on('will-frame-navigate',(event)=>{if(!allowed(event.url))event.preventDefault();});
  win.webContents.on('paint',(_event,dirty,image)=>{
    const paintNs=profileTimings?nowNs():null;
    serial++;
    const paintWall=Date.now()/1000;
    for (const socket of clients) {
      if (!socket.frameRequest || serial<=socket.frameRequest.after) continue;
      const dimensions=image.getSize();
      if (dimensions.width!==width || dimensions.height!==height) {
        cancel(socket);reply(socket,{error:'Offscreen frame geometry changed'});continue;
      }
      // toBitmap is full-frame BGRA on Windows; tests check channels and corners.
      const request=socket.frameRequest;
      const bitmapStart=profileTimings?nowNs():null;
      const pixels=image.toBitmap();
      const bitmapEnd=profileTimings?nowNs():null;
      cancel(socket);
      if (pixels.length!==width*height*4) {reply(socket,{error:'Unexpected bitmap length'});continue;}
      const metadata={width,height,channels:4,format:'BGRA',serial,paint_wall:paintWall,
        byte_length:pixels.length,dirty,clock_kind:'Electron paint callback, not GPU presentation'};
      if(shared) {
        const copyStart=profileTimings?nowNs():null;
        shared.publish(pixels,serial,width,height);
        metadata.transport='shared_memory';
        metadata.shared_copy_ms=profileTimings?elapsedMs(nowNs(),copyStart):null;
      }
      if(profileTimings)metadata.profile={
        node_request_to_paint_ms:elapsedMs(paintNs,request.receivedNs),
        bitmap_extract_ms:elapsedMs(bitmapEnd,bitmapStart),
        node_request_to_send_ms:elapsedMs(nowNs(),request.receivedNs)};
      reply(socket,metadata,shared?null:pixels);
    }
  });
  server=net.createServer(socket=>{
    console.log('raw-client-connected');
    if(clients.size) {socket.destroy();return;} // One worker, one consumer.
    clients.add(socket);socket.setNoDelay(true);
    let input='';
    socket.on('data',chunk=>{
      input+=chunk.toString('utf8');
      if(input.length>4096) {socket.destroy();return;}
      let newline;
      while((newline=input.indexOf('\n'))>=0) {
        const line=input.slice(0,newline);input=input.slice(newline+1);
        let request;try {request=JSON.parse(line);}catch {socket.destroy();return;}
        if(request.token!==args.token) {socket.destroy();return;}
        socket.authenticated=true;
        if(request.op==='close') {cancel(socket);socket.end();app.quit();return;}
        if(request.op!=='frame' || socket.frameRequest) {socket.destroy();return;}
        // A request can only receive a future paint; there is no image FIFO.
        socket.frameRequest={after:serial,receivedNs:profileTimings?nowNs():null};
        socket.frameRequest.invalidate=setTimeout(()=>win.webContents.invalidate(),80);
        socket.frameRequest.timer=setTimeout(()=>{
          cancel(socket);reply(socket,{error:'No fresh offscreen paint within 500ms'});
        },500);
      }
    });
    socket.on('error',()=>{});
    socket.on('close',()=>{
      cancel(socket);clients.delete(socket);
      if(socket.authenticated && !quitting)app.quit();
    });
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  await win.loadURL('about:blank');
  fs.writeFileSync(args.ready+'.partial',JSON.stringify({port:server.address().port,debug_port:Number(args['cdp-port']),
    electron:process.versions.electron,chrome:process.versions.chrome,width,height,
    gpu:app.getGPUFeatureStatus(),visible:win.isVisible(),offscreen:win.webContents.isOffscreen(),
    audio_muted:win.webContents.isAudioMuted(),transport:shared?"shared_memory":"socket",
    shared_name:shared?shared.name:null}));
  fs.renameSync(args.ready+'.partial',args.ready);
}).catch(error=>{console.error(error);app.exit(1);});
app.on('window-all-closed',()=>app.quit());
app.on('before-quit',()=>{quitting=true;if(shared)shared.close();for(const socket of clients){cancel(socket);socket.destroy();}if(server)server.close();});

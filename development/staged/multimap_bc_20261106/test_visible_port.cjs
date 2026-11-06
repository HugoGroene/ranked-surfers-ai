'use strict';
// Execute actual startup validation/navigation policy without launching Electron.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const file=path.join(__dirname,'../history_training_compare_20261103/visible_host.cjs');
const source=fs.readFileSync(file,'utf8');
const end=source.indexOf('async function capture(');
assert.ok(end>0);
function config(port){
  const context={URL,Buffer,process:{argv:['electron','host',`--local-port=${port}`,
    '--width=1920','--height=945','--profile=test','--ready=test','--token=test','--cdp-port=9999']},
    require(name){
      if(name==='electron')return {app:{setPath(){},commandLine:{appendSwitch(){}}}};
      if(name.endsWith('shared_frame.cjs'))return ()=>({});
      return require(name);
    }};
  vm.createContext(context);vm.runInContext(source.slice(0,end),context);return context.allowed;
}
for(const port of ['8000','8010','8012']){
  const allowed=config(port);
  assert.equal(allowed(`http://127.0.0.1:${port}/arena.html`),true);
  assert.equal(allowed(`http://localhost:${port}/`),true);
  assert.equal(allowed('about:blank'),true);
  assert.equal(allowed('https://rankedsurfers.com/'),false);
  assert.equal(allowed('http://127.0.0.1:8999/'),false);
}
assert.throws(()=>config('8999'),/Unsupported local port/);
console.log('PASS: existing map ports and BC port; navigation stays local and port-specific.');

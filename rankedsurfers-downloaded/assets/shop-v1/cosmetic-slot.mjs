// One canonical back slot. Conflicting legacy data resolves to one item, never layers.
export const BACK_IDS=Object.freeze(['none','cape','wings','eclipse']);
export function cosmeticBack(value={}){
 if(Object.hasOwn(value||{},'back'))return BACK_IDS.includes(value.back)?value.back:'none';
 if(value?.accessory==='none')return 'none';
 if(value?.accessory==='cape'||value?.cape)return 'cape';
 if(value?.accessory==='wings'||value?.wing)return (value.wingDesign||value.wing)==='eclipse'?'eclipse':'wings';
 return 'none';
}
export function avatarEquipment(value){
 const back=cosmeticBack(value);
 return back==='cape'?{parts:['cape'],cape:'classic'}:back==='none'?{parts:[]}:
 {parts:['wings'],wing:'classic',wingStyle:back==='eclipse'?6:0};
}
export function assertOwnedBack(back,owned){
 if(!BACK_IDS.includes(back))throw Error('Invalid back accessory.');
 if(back!=='none'&&!owned.includes(back))throw Error('This accessory is not owned.');
 return back;
}

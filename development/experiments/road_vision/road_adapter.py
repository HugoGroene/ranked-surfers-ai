"""Process-local composition with existing temporal/settled learning fixes."""
from road_config import ROOT

def install(trainer,config,*,crop_previews=False):
    from menu_confirmation import install as install_confirmation
    install_confirmation()
    import live_adapter
    from history_spacing import history_class
    live_adapter.ContextHistory=history_class(config['temporal'].get('spacing','uniform'))
    import tools.ppo_framing as framing
    import tools.ppo_observation_contract as contracts
    from road_crop import observation,preview
    def policy_crop(image,target=(512,288),interpolation=None):
        return observation(image,config)
    framing.canvas_observation=policy_crop
    live_adapter.install(trainer,config['temporal'])
    previous_contract=contracts.requested_contract
    def requested(a):
        c=previous_contract(a);w,h=config['size']
        c.update(image_shape=[3*config['temporal']['frames'],h,w],road_vision=dict(config),
                 resize_method='area')
        return c
    contracts.requested_contract=requested
    previous_build=trainer.build_env_class
    def build():
        base=previous_build()
        class RoadEnv(base):
            def __init__(self,*a,**kw):
                self._road_preview_saved=False
                self._road_gameplay_previews=set()
                super().__init__(*a,**kw)
            def _native_frames(self):
                small,native=super()._native_frames()
                if not self._road_preview_saved:
                    preview(native,config,self.output_dir/'road_crop_bounds.png',
                            writer=getattr(self,'_policy_preview_writer',None))
                    from road_config import write_json
                    write_json(self.output_dir/'road_crop_manifest.json',dict(config=config,
                        guard_view='unchanged full canvas',candidate_not_visibility_validated=True))
                    profile_path=self.output_dir/'framing_profile.json'
                    if profile_path.exists():
                        import json
                        profile=json.loads(profile_path.read_text())
                        profile.update(full_canvas=config['crop']=='full',native_full_canvas=True,
                                       policy_roi=config['roi'],road_vision=config)
                        write_json(profile_path,profile)
                    self._road_preview_saved=True
                return small,native
            def _pack(self,*a,**kw):
                result=super()._pack(*a,**kw)
                step=getattr(self,'training_steps',0)
                if crop_previews and step in (10,31,67,109,157,211,269,331,401,479) and step not in self._road_gameplay_previews:
                    import cv2
                    from road_crop import crop_rgb
                    native=self.last_source_native
                    preview(native,config,self.output_dir/f'road_bounds_step{step}.png')
                    small=crop_rgb(native,config).transpose(1,2,0)
                    cv2.imwrite(str(self.output_dir/f'road_input_step{step}.png'),cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
                    self._road_gameplay_previews.add(step)
                return result
        return RoadEnv
    trainer.build_env_class=build
    previous_make=trainer.make_model
    def make(env,args):
        if config['encoder']=='tiny':return previous_make(env,args)
        import train_rs_v3_ppo as v3
        from road_models import SpatialExtractor
        old=v3.RGBHazardExtractor
        try:
            v3.RGBHazardExtractor=SpatialExtractor
            return previous_make(env,args)
        finally:v3.RGBHazardExtractor=old
    trainer.make_model=make

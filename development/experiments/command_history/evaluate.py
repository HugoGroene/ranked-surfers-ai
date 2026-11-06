"""Matching evaluation entry point; accepts evaluate_model.py's arguments."""
from command_adapter import install
import os

if __name__=='__main__':
    trainer=install()
    if os.environ.get('RS_DEFER_STARTUP_PREVIEWS')=='1':
        from development.experiments.multimap_classifier.deferred_previews import install as install_previews
        install_previews(trainer)
    if os.environ.get('RS_REUSE_PUMP_CROP')=='1':
        from development.experiments.multimap_classifier.crop_reuse import install as install_crop
        install_crop(trainer)
    from development.experiments.multimap_classifier.eval_latency import install as install_eval_latency
    install_eval_latency(trainer)
    from evaluate_model import main
    main()

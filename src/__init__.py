try:
    from catalyst.dl import SupervisedRunner as Runner
    from .experiment import Experiment
    from .model import MNISTNet
    from catalyst.dl import registry
    from .callbacks.infer_callback import MNISTInferCallback

    registry.Model(MNISTNet)
except ImportError:
    print("Catalyst not found. Loading production environment")

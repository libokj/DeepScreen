from omegaconf import OmegaConf
# Register a new resolver to support arithmetic operations in variable interpolation in YAML configuration files
OmegaConf.register_new_resolver("eval", eval)

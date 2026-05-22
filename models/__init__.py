# ./models/__init__.py

from .SALMONN3D import SALMONN3D

def load_model(model_cfg):
	"""Factory function used by train.py to construct the model.

	Args:
		model_cfg: cfg.config.model node from the YAML config. It is passed
			directly to SALMONN3D.from_config, which handles keys like
			`qwen_path`, `ckpt`, and LoRA-related settings.

	Returns:
		An instance of SALMONN3D ready for training or evaluation.
	"""

	model = SALMONN3D.from_config(model_cfg)
	return model
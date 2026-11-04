from typing import Tuple, Optional, Any

try:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    HAS_TORCH = True
except ImportError:
    torch = None
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    HAS_TORCH = False

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

def load_qwen_vl(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    dtype: Optional[Any] = None
) -> Tuple[Any, Any, Any]:
    """
    Load Qwen2.5-VL processor and model with optimal device placement.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch and transformers must be installed to load Qwen2.5-VL models.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
        
    print(f"[ModelLoader] Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f"[ModelLoader] Loading weights for {model_id} on {device} ({dtype})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        model = model.to(device)
        
    target_device = next(model.parameters()).device
    print(f"[ModelLoader] Model ready on device: {target_device}")
    return processor, model, target_device

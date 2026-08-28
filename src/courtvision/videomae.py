"""Loading VideoMAE with its attention biases intact.

MCG-NJU's VideoMAE checkpoints store attention biases BEiT-style, as `q_bias`
and `v_bias` per layer, with no key bias (that is the architecture: key bias is
genuinely absent, not missing). transformers >= 5 renamed these to
`query.bias` / `key.bias` / `value.bias` and does NOT map the old names across,
so `from_pretrained` reports them UNEXPECTED/MISSING and silently leaves all 24
loaded biases at zero.

Nothing errors; the model just runs with part of its pretrained attention
discarded. This restores them.
"""

from __future__ import annotations


def load_videomae_classifier(model_id: str, **kwargs):
    """Load VideoMAEForVideoClassification with q_bias/v_bias restored.

    Returns (model, restored_count). A restored_count of 0 means either the
    checkpoint already used the new names or transformers started mapping them —
    both fine, and worth noticing rather than assuming.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import VideoMAEForVideoClassification

    model = VideoMAEForVideoClassification.from_pretrained(model_id, **kwargs)

    try:
        checkpoint = load_file(hf_hub_download(model_id, "model.safetensors"))
    except Exception:
        # A locally fine-tuned checkpoint we saved ourselves already has the
        # correct names, so there is nothing to restore.
        return model, 0

    params = dict(model.named_parameters())
    restored = 0
    with torch.no_grad():
        for key, value in checkpoint.items():
            if key.endswith(".q_bias"):
                target = key.replace(".q_bias", ".query.bias")
            elif key.endswith(".v_bias"):
                target = key.replace(".v_bias", ".value.bias")
            else:
                continue
            param = params.get(target)
            if param is not None and param.shape == value.shape:
                param.copy_(value)
                restored += 1
    return model, restored

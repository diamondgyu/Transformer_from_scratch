import pathlib
import sys
import torch


def cut_string_between_bos_eos(string, bos='[CLS]', eos='[SEP]'):
    """Extract string content between BOS and EOS tokens."""
    bos_index = string.find(bos)
    eos_index = string.find(eos)
    if bos_index == -1:
        bos_index = -len(bos)
    if eos_index == -1:
        eos_index = len(string)
    return string[bos_index + len(bos):eos_index].strip()


def load_model(model_pt_path: pathlib.Path, config: dict):
    """Reconstruct TransformerModel, load weights in bf16, return eval model."""
    # Import lazily so the module works even without torch installed at import time.
    src_dir = str(pathlib.Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from model import TransformerModel  # type: ignore

    model = TransformerModel(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        dropout_rate=config["dropout_rate"],
        hidden_layer_dim=config["hidden_layer_dim"],
        max_len=config["max_len"],
        vocab_size=config["vocab_size"],
        stacks=config["stacks"],
        pad_token_id=config["pad_token_id"],
        rope_base=config["rope_base"],
    )

    state_dict = torch.load(str(model_pt_path), weights_only=False, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(torch.bfloat16)
    model.eval()
    return model


def beam_generate(model, src_ids: torch.Tensor, num_beams: int = 3,
                  max_len: int = 128, length_penalty: float = 1.0,
                  repetition_penalty: float = 1.3) -> torch.Tensor:
    """Thin wrapper: delegates to model.beam_generate() which has full KV-cache beam search."""
    return model.beam_generate(
        src_ids,
        num_beams=num_beams,
        max_len=max_len,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
    )

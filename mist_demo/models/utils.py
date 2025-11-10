import logging
from pathlib import Path

def save_model(model, save_directory, safe_serialization=False):
    if safe_serialization:
        from safetensors.torch import save_model

        save_model(model, Path(save_directory, "model.safetensors"))
    else:
        torch.save(model.state_dict(), Path(save_directory, "model.pt"))


def load_model(model, save_directory):
    if (file := Path(save_directory, "model.safetensors")).is_file():
        from safetensors.torch import load_model as st_load_model

        unexpected, missing = st_load_model(model, file)
        if unexpected or missing:
            logging.warning(
                "Unexpected or missing tensors when loading %s, unexpected: %s missing: %s",
                file,
                unexpected,
                missing,
            )

    elif (file := Path(save_directory, "model.pt")).is_file():
        model.load_state_dict(torch.load(file, weights_only=True))
    else:
        raise RuntimeError("No model found")
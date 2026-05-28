import logging

import torch


def select_device(logger: logging.Logger) -> str:
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("CUDA is available. Using GPU: %s", gpu_name)
        except Exception as exc:
            logger.warning("CUDA initialization warning: %s", exc)
        return "cuda"

    diagnostics: list[str] = []
    try:
        if hasattr(torch.backends, "cuda") and not torch.backends.cuda.is_built():
            diagnostics.append("Current PyTorch build lacks CUDA support.")
    except Exception as exc:
        diagnostics.append(f"Could not query torch.backends.cuda: {exc}")

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        diagnostics.append(
            "torch.version.cuda is None. Install GPU-enabled wheels, "
            "for example: pip install --upgrade torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        diagnostics.append(f"PyTorch reports CUDA runtime {cuda_version}.")

    try:
        gpu_count = torch.cuda.device_count()
        diagnostics.append(f"Detected CUDA devices: {gpu_count}")
        if gpu_count == 0:
            diagnostics.append(
                "No CUDA-capable GPUs detected. Ensure NVIDIA drivers are installed "
                "and `nvidia-smi` works."
            )
    except Exception as exc:
        diagnostics.append(f"Could not enumerate CUDA devices: {exc}")

    logger.warning("CUDA not available, defaulting to CPU.")
    for line in diagnostics:
        logger.warning("Diagnostics: %s", line)
    return "cpu"

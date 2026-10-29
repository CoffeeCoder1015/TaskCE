"""Pack sample masks before transferring them to the search device."""

import numpy as np
import torch


def pack_vectors(vectors: np.ndarray, device: torch.device) -> torch.Tensor:
    """Pack [vector, sample] into int32 [vector, word], with zero tail padding."""
    packed_bytes = np.packbits(vectors, axis=1, bitorder="little")
    padding_bytes = -packed_bytes.shape[1] % 4
    if padding_bytes:
        packed_bytes = np.pad(packed_bytes, ((0, 0), (0, padding_bytes)))
    packed_words = np.ascontiguousarray(packed_bytes).view(np.int32)
    return torch.from_numpy(packed_words).to(device=device)

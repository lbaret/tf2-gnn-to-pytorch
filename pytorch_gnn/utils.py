from typing import Callable, Optional

import torch
from torch_scatter import segment_coo


def get_activation_function_by_name(
    activation_fn_name: Optional[str],
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Convert from an activation function name to the function itself."""
    if activation_fn_name is None:
        return None
    activation_fn_name = activation_fn_name.lower()

    string_to_activation_fn = {
        "linear": None,
        "tanh": torch.nn.Tanh(),
        "relu": torch.nn.ReLU(),
        "leaky_relu": torch.nn.LeakyReLU(),
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "gelu": torch.nn.GELU(),
    }
    activation_fn = string_to_activation_fn.get(activation_fn_name)
    if activation_fn is None:
        raise ValueError(f"Unknown activation function: {activation_fn_name}")
    return activation_fn

# TODO: Test this function and compare results with TF version
def unsorted_segment_softmax(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int=1) -> torch.Tensor:
    """Perform a safe unsorted segment softmax."""
    max_per_segment = segment_coo(
        logits, segment_ids, reduce="max"
    )
    scattered_maxes = max_per_segment[segment_ids] # Gather over dim=0 and providing only array of IDs is equivalent to getting item in torch
    recentered_scores = logits - scattered_maxes
    exped_recentered_scores = torch.exp(recentered_scores)

    per_segment_sums = segment_coo(
        exped_recentered_scores, segment_ids, reduce="sum"
    )

    probs = exped_recentered_scores / (
        per_segment_sums[segment_ids] + 1e-7
    )
    return probs

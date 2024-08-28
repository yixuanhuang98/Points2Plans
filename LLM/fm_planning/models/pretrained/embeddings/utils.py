import torch


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform mean pooling on the model output.

    Args:
        model_output: Output from the HuggingFace model.
        attention_mask (torch.Tensor): Attention mask tensor.

    Returns:
        torch.Tensor: Mean pooled tensor.
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    mean_pool = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return mean_pool

from typing import Optional

import torch


def mask_input_ids(
    input_ids: torch.Tensor,
    special_token_ids: list[int],
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT-style masking: of the selected tokens, 80% become [MASK], 10% a random token, 10% stay.

    Returns (masked input ids, labels) with labels set to -100 at unselected positions.
    """
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    shape = input_ids.shape

    special = torch.isin(input_ids, torch.tensor(special_token_ids))
    selection_probability = torch.full(shape, mlm_probability)
    selection_probability[special] = 0.0
    selected = torch.bernoulli(selection_probability, generator=generator).bool()
    labels[~selected] = -100

    use_mask_token = torch.bernoulli(torch.full(shape, 0.8), generator=generator).bool() & selected
    input_ids[use_mask_token] = mask_token_id

    use_random_token = torch.bernoulli(torch.full(shape, 0.5), generator=generator).bool() & selected & ~use_mask_token
    random_tokens = torch.randint(vocab_size, shape, generator=generator)
    input_ids[use_random_token] = random_tokens[use_random_token]
    return input_ids, labels

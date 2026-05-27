import random
from typing import List

from .config import QueryGenerationConfig
from .types import ChunkedDocument


def choose_query_types_for_doc(
    doc: ChunkedDocument,
    config: QueryGenerationConfig,
) -> List[str]:
    if config.query_types_per_doc >= len(config.query_types):
        return list(config.query_types)

    seed = f"{config.query_type_sample_seed}:{doc.doc_id}"
    rng = random.Random(seed)
    chosen = rng.sample(list(config.query_types), config.query_types_per_doc)
    return [query_type for query_type in config.query_types if query_type in chosen]


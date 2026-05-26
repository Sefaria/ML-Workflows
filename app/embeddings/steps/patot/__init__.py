from .config import ChunkerConfig
from .json_loader import load_segment_records_from_json_file, load_segment_records_from_section
from .records import SegmentRecord

try:
    from .chunker import PatotChunk, PatotChunkResult, PatotChunker
except ModuleNotFoundError:
    # Allow JSON loading utilities to be imported before optional chunking deps are installed.
    PatotChunker = None
    PatotChunk = None
    PatotChunkResult = None

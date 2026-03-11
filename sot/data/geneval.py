"""GenEval dataset loader with auto-download from GitHub.

GenEval is an object-focused framework for evaluating text-to-image alignment.
Paper: https://arxiv.org/abs/2310.11513
Repo: https://github.com/djghosh13/geneval
"""

import json
import urllib.request
from pathlib import Path
from typing import Tuple, List, Dict, Optional

__all__ = ["load_geneval_captions", "download_geneval_data"]

GENEVAL_PROMPTS_URL = "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/generation_prompts.txt"
GENEVAL_METADATA_URL = "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl"


def download_geneval_data(dataset_dir: str = "datasets/geneval") -> Path:
    """Auto-download GenEval prompts and metadata if not present.
    
    Downloads data from the official GenEval GitHub repository:
    https://github.com/djghosh13/geneval
    
    Args:
        dataset_dir: Directory to download data to
        
    Returns:
        Path to prompts directory
        
    Raises:
        RuntimeError: If download fails
    """
    dataset_path = Path(dataset_dir)
    prompts_dir = dataset_path / "prompts"
    prompts_file = prompts_dir / "generation_prompts.txt"
    metadata_file = prompts_dir / "evaluation_metadata.jsonl"
    
    # Check if already downloaded
    if prompts_file.exists() and metadata_file.exists():
        return prompts_dir
    
    print(f"GenEval data not found. Downloading to {dataset_dir}...")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Download prompts
    print(f"  Downloading prompts from GitHub...")
    try:
        urllib.request.urlretrieve(GENEVAL_PROMPTS_URL, prompts_file)
        print(f"  ✓ Saved to {prompts_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to download prompts: {e}")
    
    # Download metadata
    print(f"  Downloading metadata from GitHub...")
    try:
        urllib.request.urlretrieve(GENEVAL_METADATA_URL, metadata_file)
        print(f"  ✓ Saved to {metadata_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to download metadata: {e}")
    
    print("  Download complete!")
    print(f"  GenEval repo: https://github.com/djghosh13/geneval")
    return prompts_dir


def load_geneval_captions(
    dataset_dir: str = "datasets/geneval",
    num_samples: Optional[int] = None
) -> Tuple[List[str], List[Dict]]:
    """Load GenEval captions and metadata, auto-downloading if needed.
    
    The GenEval dataset contains prompts for evaluating compositional
    capabilities of text-to-image models including:
    - Single object generation
    - Two object co-occurrence  
    - Counting
    - Colors
    - Spatial relationships
    - Color attribution
    
    Args:
        dataset_dir: Directory containing (or to download) GenEval data
        num_samples: Number of samples to load (None = all, default)
        
    Returns:
        (captions, metadata): Lists of prompts and metadata dicts.
            - captions: List of text prompts (e.g., "a red apple and a blue banana")
            - metadata: List of dicts with task info (e.g., {"task": "two_obj", "objects": ["apple", "banana"]})
    """
    # Auto-download if needed
    prompts_dir = download_geneval_data(dataset_dir)
    
    # Load prompts
    prompts_file = prompts_dir / "generation_prompts.txt"
    with open(prompts_file, 'r') as f:
        captions = [line.strip() for line in f.readlines()]
    
    # Load metadata
    metadata_file = prompts_dir / "evaluation_metadata.jsonl"
    with open(metadata_file, 'r') as f:
        metadata = [json.loads(line) for line in f.readlines()]
    
    # Optionally limit samples
    if num_samples is not None:
        captions = captions[:num_samples]
        metadata = metadata[:num_samples]
    
    print(f"Loaded {len(captions)} GenEval prompts")
    return captions, metadata


"""Download and prepare datasets for F5-TTS training"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_emilia(output_dir: str, revision: str = "fc71e07") -> None:
    """Download Emilia Dataset from Hugging Face"""
    print("\nDownloading Emilia Dataset...")
    
    # Download from HF hub
    cache_dir = snapshot_download(
        repo_id="amphion/Emilia-Dataset",
        revision=revision,
        repo_type="dataset"
    )

    # Create output structure
    output_path = Path(output_dir) / "Emilia_Dataset"
    output_path.mkdir(parents=True, exist_ok=True)
    
    raw_dir = output_path / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Copy files preserving structure
    for lang in ["EN", "ZH"]:
        print(f"\nCopying {lang} files...")
        src = Path(cache_dir) / lang
        dst = raw_dir / lang
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    print(f"\nEmilia Dataset downloaded to: {output_path}")


def download_libritts(output_dir: str) -> None:
    """Download LibriTTS dataset using wget"""
    print("\nDownloading LibriTTS...")
    
    output_path = Path(output_dir) / "LibriTTS"
    output_path.mkdir(parents=True, exist_ok=True)

    subsets = [
        "train-clean-100",
        "train-clean-360", 
        "train-other-500"
    ]

    base_url = "https://www.openslr.org/resources/60"
    
    for subset in subsets:
        tar_file = f"{subset}.tar.gz"
        url = f"{base_url}/{tar_file}"
        
        print(f"\nDownloading {subset}...")
        subprocess.run(["wget", url, "-P", str(output_path)], check=True)
        
        print(f"Extracting {subset}...")
        subprocess.run(
            ["tar", "xzf", str(output_path/tar_file), "-C", str(output_path)],
            check=True
        )
        
        # Cleanup tar file
        (output_path/tar_file).unlink()

    print(f"\nLibriTTS downloaded to: {output_path}")


def main(output_dir: Optional[str] = None):
    """Main download function"""
    if output_dir is None:
        output_dir = os.path.expanduser("~/datasets")
    
    print(f"\nDatasets will be downloaded to: {output_dir}")
    
    download_emilia(output_dir)
    download_libritts(output_dir)


if __name__ == "__main__":
    main()

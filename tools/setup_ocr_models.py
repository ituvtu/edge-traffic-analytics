import os
import sys
import shutil
import tarfile
import subprocess
import urllib.request
from pathlib import Path

# Configuration
PADDLE_DET_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
PADDLE_REC_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"
# Standard English dictionary for PaddleOCR
PADDLE_DICT_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt"

PROJECT_ROOT = Path(__file__).parent.parent
ONNX_DIR = PROJECT_ROOT / "onnx"
TEMP_DIR = PROJECT_ROOT / "temp_ocr_download"

def get_paddle2onnx_cmd():
    """Find the paddle2onnx executable."""
    # Algorithm 1: Check if 'paddle2onnx' is in PATH (but prefer full path if not working)
    if shutil.which("paddle2onnx"):
        # Double check if it actually runs. Sometimes 'which' finds it but permission/execution fails
        try:
             subprocess.run(["paddle2onnx", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             return "paddle2onnx"
        except:
             pass

    # Algorithm 2: Check relative to the current python interpreter
    python_dir = Path(sys.executable).parent
    candidates = [
        python_dir / "Scripts" / "paddle2onnx.exe",
        python_dir / "Scripts" / "paddle2onnx",
        python_dir / "bin" / "paddle2onnx",
        python_dir / "paddle2onnx.exe",
        python_dir / "paddle2onnx"
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Algorithm 3: Fallback check for "paddle2onnx.exe" if user is on Windows but didn't put Scripts in PATH
    if os.name == 'nt':
         if shutil.which("paddle2onnx.exe"):
             return "paddle2onnx.exe"

    return None

def check_paddle2onnx():
    """Check if paddle2onnx is installed and available in PATH."""
    cmd = get_paddle2onnx_cmd()
    if not cmd:
        print("[ERROR] paddle2onnx executable not found.")
        print(f"Current Python: {sys.executable}")
        print("Please ensure it is installed: pip install paddle2onnx")
        sys.exit(1)
        
    try:
        subprocess.run([cmd, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[INFO] paddle2onnx is installed (using: {cmd}).")
        return cmd
    except subprocess.CalledProcessError:
         # Check if it failed due to missing paddle
        try:
             result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
             if "No module named 'paddle'" in result.stderr or "Failed to import paddle" in result.stderr:
                 print("[ERROR] paddle2onnx requires paddlepaddle to function.")
                 print("Please install it: pip install paddlepaddle")
                 sys.exit(1)
        except:
             pass
        print(f"[ERROR] Failed to run found command: {cmd}.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"[ERROR] Command not found: {cmd}.")
        sys.exit(1)

# ... (rest of imports)


def download_file(url: str, dest_path: Path):
    """Download a file from a URL to a destination path."""
    print(f"[INFO] Downloading {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"[INFO] Downloaded to {dest_path}")

def extract_tar(tar_path: Path, extract_to: Path):
    """Extract a tar archive."""
    print(f"[INFO] Extracting {tar_path}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_to)
    print(f"[INFO] Extracted to {extract_to}")

def convert_to_onnx(model_dir: Path, save_file: Path, paddle_cmd: str):
    """Convert PaddlePaddle model to ONNX using paddle2onnx CLI."""
    # Check for both standard naming conventions
    possible_names = [
        ("inference.pdmodel", "inference.pdiparams"),
        ("model.pdmodel", "model.pdiparams")
    ]
    
    model_file = None
    params_file = None
    
    for m_name, p_name in possible_names:
        if (model_dir / m_name).exists() and (model_dir / p_name).exists():
            model_file = m_name
            params_file = p_name
            break
            
    if not model_file:
        raise FileNotFoundError(f"Model files not found in {model_dir}. Checked for inference.pdmodel/model.pdmodel")

    print(f"[INFO] Converting model in {model_dir} to {save_file}...")
    
    # Ensure output directory exists
    save_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(paddle_cmd),
        "--model_dir", str(model_dir),
        "--model_filename", model_file,
        "--params_filename", params_file,
        "--save_file", str(save_file),
        "--opset_version", "11",
        "--enable_onnx_checker", "True"
    ]
    
    subprocess.run(cmd, check=True)
    print(f"[SUCCESS] Converted to {save_file}")

def download_and_save_keys(url: str, save_path: Path):
    """Download the character dictionary."""
    print(f"[INFO] Downloading dictionary from {url}...")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, save_path)
    print(f"[INFO] Saved dictionary to {save_path}")

def cleanup():
    """Remove temporary files and directories."""
    if TEMP_DIR.exists():
        print(f"[INFO] Cleaning up {TEMP_DIR}...")
        shutil.rmtree(TEMP_DIR)
        print("[INFO] Cleanup complete.")

def main():
    paddle_cmd = check_paddle2onnx()
    
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Detection Model
        det_tar = TEMP_DIR / "det.tar"
        download_file(PADDLE_DET_URL, det_tar)
        extract_tar(det_tar, TEMP_DIR)
        # The tar usually extracts to a subdirectory named after the tar file minus extension
        det_model_dir = TEMP_DIR / "en_PP-OCRv3_det_infer"
        convert_to_onnx(det_model_dir, ONNX_DIR / "ocr_det.onnx", paddle_cmd)

        # 2. Recognition Model
        rec_tar = TEMP_DIR / "rec.tar"
        download_file(PADDLE_REC_URL, rec_tar)
        extract_tar(rec_tar, TEMP_DIR)
        rec_model_dir = TEMP_DIR / "en_PP-OCRv4_rec_infer"
        convert_to_onnx(rec_model_dir, ONNX_DIR / "ocr_rec.onnx", paddle_cmd)

        # 3. Character Dictionary
        download_and_save_keys(PADDLE_DICT_URL, ONNX_DIR / "ocr_keys.txt")

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        # Clean up even on error if desired, or keep for debugging. 
        # Requirement says "final block ... for removal", implying success path mostly.
        # But try/finally is safer.
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()

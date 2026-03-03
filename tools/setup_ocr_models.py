import sys
import shutil
import tarfile
import subprocess
import urllib.request
from pathlib import Path

PADDLE_DET_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
PADDLE_REC_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"
PADDLE_DICT_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt"

ROOT = Path(__file__).parent.parent
ONNX = ROOT / "onnx"
TEMP = ROOT / "temp_ocr_download"

def find_paddle2onnx():
    exe = shutil.which("paddle2onnx")
    if exe:
        try:
            subprocess.run([exe, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return exe
        except Exception:
            pass
    py_dir = Path(sys.executable).parent
    for path in [
        py_dir / "Scripts" / "paddle2onnx.exe",
        py_dir / "Scripts" / "paddle2onnx",
        py_dir / "bin" / "paddle2onnx",
        py_dir / "paddle2onnx.exe",
        py_dir / "paddle2onnx"
    ]:
        if path.exists():
            return str(path)
    if sys.platform == "win32":
        exe = shutil.which("paddle2onnx.exe")
        if exe:
            return exe
    return None

def require_paddle2onnx():
    exe = find_paddle2onnx()
    if not exe:
        sys.exit("paddle2onnx not found. Install: pip install paddle2onnx")
    try:
        subprocess.run([exe, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return exe
    except subprocess.CalledProcessError:
        result = subprocess.run([exe, "--version"], capture_output=True, text=True)
        if "No module named 'paddle'" in result.stderr or "Failed to import paddle" in result.stderr:
            sys.exit("paddlepaddle required. Install: pip install paddlepaddle")
        sys.exit(f"Failed to run: {exe}")
    except FileNotFoundError:
        sys.exit(f"Command not found: {exe}")

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)

def untar(src, dst):
    with tarfile.open(src, "r") as tar:
        tar.extractall(path=dst)

def to_onnx(model_dir, out_file, paddle2onnx):
    for m, p in [("inference.pdmodel", "inference.pdiparams"), ("model.pdmodel", "model.pdiparams")]:
        if (model_dir / m).exists() and (model_dir / p).exists():
            cmd = [
                paddle2onnx,
                "--model_dir", str(model_dir),
                "--model_filename", m,
                "--params_filename", p,
                "--save_file", str(out_file),
                "--opset_version", "11",
                "--enable_onnx_checker", "True"
            ]
            out_file.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)
            return
    raise FileNotFoundError(f"Model files not found in {model_dir}")

def clean():
    if TEMP.exists():
        shutil.rmtree(TEMP)

def main():
    paddle2onnx = require_paddle2onnx()
    ONNX.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    try:
        det_tar = TEMP / "det.tar"
        download(PADDLE_DET_URL, det_tar)
        untar(det_tar, TEMP)
        to_onnx(TEMP / "en_PP-OCRv3_det_infer", ONNX / "ocr_det.onnx", paddle2onnx)

        rec_tar = TEMP / "rec.tar"
        download(PADDLE_REC_URL, rec_tar)
        untar(rec_tar, TEMP)
        to_onnx(TEMP / "en_PP-OCRv4_rec_infer", ONNX / "ocr_rec.onnx", paddle2onnx)

        download(PADDLE_DICT_URL, ONNX / "ocr_keys.txt")
    finally:
        clean()

if __name__ == "__main__":
    main()

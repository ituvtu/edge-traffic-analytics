from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO


def export_model(pt_path: Path, onnx_path: Path, opset: int) -> Path:
    if not pt_path.exists():
        raise FileNotFoundError(f"PT model not found: {pt_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    exported = Path(YOLO(str(pt_path)).export(format="onnx", opset=opset, imgsz=640))
    if exported.resolve() != onnx_path.resolve():
        exported.replace(onnx_path)
    return onnx_path


def dry_run_load(model_path: Path, providers: list[str]) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found for dry-run: {model_path}")
    ort.InferenceSession(str(model_path), providers=providers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLOv10 Nano PT models to ONNX")
    parser.add_argument("--car-pt", type=Path, default=Path("pt/yolo10nano_car.pt"))
    parser.add_argument("--plate-pt", type=Path, default=Path("pt/yolo10nano_plate.pt"))
    parser.add_argument("--car-onnx", type=Path, default=Path("onnx/yolo10nano_car.onnx"))
    parser.add_argument("--plate-onnx", type=Path, default=Path("onnx/yolo10nano_plate.onnx"))
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--skip-export", action="store_true", help="Skip export and only run provider dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Load ONNX with CPU and CUDA providers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_export:
        export_model(args.car_pt, args.car_onnx, args.opset)
        export_model(args.plate_pt, args.plate_onnx, args.opset)

    if args.dry_run:
        dry_run_load(args.car_onnx, ["CPUExecutionProvider"])
        dry_run_load(args.plate_onnx, ["CPUExecutionProvider"])
        if "CUDAExecutionProvider" in ort.get_available_providers():
            dry_run_load(args.car_onnx, ["CUDAExecutionProvider", "CPUExecutionProvider"])
            dry_run_load(args.plate_onnx, ["CUDAExecutionProvider", "CPUExecutionProvider"])


if __name__ == "__main__":
    main()

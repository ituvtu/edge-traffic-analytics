from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RuntimeProfile:
    name: str
    providers: list[str]


@dataclass(slots=True)
class ModelPathsConfig:
    car_model_path: Path = Path(os.getenv("CAR_MODEL_PATH", "onnx/yolo10nano_car.onnx"))
    plate_model_path: Path = Path(os.getenv("PLATE_MODEL_PATH", "onnx/yolo10nano_plate.onnx"))
    ocr_det_path: Path = Path(os.getenv("OCR_DET_PATH", "onnx/ocr_det.onnx"))
    ocr_rec_path: Path = Path(os.getenv("OCR_REC_PATH", "onnx/ocr_rec.onnx"))
    ocr_char_dict_path: Path = Path(os.getenv("OCR_CHAR_DICT_PATH", "onnx/ocr_keys.txt"))


@dataclass(slots=True)
class ThresholdsConfig:
    vehicle_conf_threshold: float = float(os.getenv("VEHICLE_CONF_THRESHOLD", "0.20"))
    plate_conf_threshold: float = float(os.getenv("PLATE_CONF_THRESHOLD", "0.30"))
    nms_threshold: float = float(os.getenv("NMS_THRESHOLD", "0.45"))
    ocr_min_vehicle_width: int = int(os.getenv("OCR_MIN_VEHICLE_WIDTH", "400"))
    ocr_min_vehicle_area: int = int(os.getenv("OCR_MIN_VEHICLE_AREA", "120000"))


def _default_ocr_regex_patterns() -> list[str]:
    env_patterns = os.getenv("OCR_REGEX_PATTERNS")
    if env_patterns:
        parsed = [pattern.strip() for pattern in env_patterns.split(";") if pattern.strip()]
        if parsed:
            return parsed
    return [
        # Ukrainian:  АА1234ВВ, АА123456
        r"^[A-ZА-ЯІЇЄ]{1,3}[0-9]{3,5}[A-ZА-ЯІЇЄ]{0,3}$",
        # Turkish:    34NOC830, 06ABC123
        r"^[0-9]{2}[A-Z]{1,3}[0-9]{2,5}$",
        # European:   AB1234CD, 1234ABC
        r"^[A-Z0-9]{4,10}$",
    ]


@dataclass(slots=True)
class OCRRegexConfig:
    patterns: list[str] = field(default_factory=_default_ocr_regex_patterns)


def _default_rtsp_sources() -> dict[str, str]:
    rtsp_sources_env = os.getenv("RTSP_URLS", "").strip()
    if rtsp_sources_env:
        parsed: dict[str, str] = {}
        for item in rtsp_sources_env.split(";"):
            key, _, value = item.partition("=")
            key = key.strip()
            value = value.strip()
            if key and value:
                parsed[key] = value
        if parsed:
            return parsed
    return {"main": os.getenv("RTSP_URL", "")}


def _next_artifact_path(base: str, ext: str) -> Path:
    folder = Path("artifacts")
    folder.mkdir(exist_ok=True)
    existing = sorted(folder.glob(f"{base}_*.{ext}"))
    if not existing:
        return folder / f"{base}_1.{ext}"
    nums = [int(p.stem.split('_')[-1]) for p in existing if p.stem.split('_')[-1].isdigit()]
    next_num = max(nums, default=0) + 1
    return folder / f"{base}_{next_num}.{ext}"


@dataclass(slots=True)
class AppConfig:
    rtsp_sources: dict[str, str] = field(default_factory=_default_rtsp_sources)
    default_rtsp_source: str = os.getenv("DEFAULT_RTSP_SOURCE", "main")
    models: ModelPathsConfig = field(default_factory=ModelPathsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    ocr_regex: OCRRegexConfig = field(default_factory=OCRRegexConfig)
    output_csv_path: Path = Path(os.getenv("OUTPUT_CSV_PATH", "artifacts/results.csv"))
    report_csv_path: Path = Path(os.getenv("REPORT_CSV_PATH", "artifacts/report.csv"))
    profiler_report_path: Path = Path("artifacts/profiler.txt")
    save_video: bool = os.getenv("SAVE_VIDEO", "false").lower() in ("1", "true", "yes")
    output_video_path: Path | None = (
        Path(os.environ["OUTPUT_VIDEO_PATH"]) if "OUTPUT_VIDEO_PATH" in os.environ else None
    )
    headless_mode: bool = os.getenv("HEADLESS_MODE", "false").lower() in ("1", "true", "yes")
    frame_skip: int = int(os.getenv("FRAME_SKIP", "1"))
    reconnect_initial_delay_sec: float = float(os.getenv("RECONNECT_INITIAL_DELAY_SEC", "1.0"))
    reconnect_max_delay_sec: float = float(os.getenv("RECONNECT_MAX_DELAY_SEC", "8.0"))
    reset_stats_on_loop: bool = os.getenv("RESET_STATS_ON_LOOP", "true").lower() in ("1", "true", "yes")
    vehicle_class_ids: tuple[int, ...] = (2, 3, 5, 7)
    input_size: tuple[int, int] = (640, 640)
    profile: str = os.getenv("PROFILE", "cpu")

    def __post_init__(self) -> None:
        if self.save_video and self.output_video_path is None:
            self.output_video_path = _next_artifact_path("output", "mp4")
        if "REPORT_CSV_PATH" not in os.environ:
            self.report_csv_path = _next_artifact_path("report", "csv")
        # Derive profiler report path from report_csv_path session number.
        # e.g. artifacts/report_3.csv  →  artifacts/profiler_3.txt
        stem = self.report_csv_path.stem          # "report_3"
        suffix = stem.split("_")[-1]              # "3"
        if suffix.isdigit():
            self.profiler_report_path = self.report_csv_path.parent / f"profiler_{suffix}.txt"
        else:
            self.profiler_report_path = self.report_csv_path.parent / "profiler.txt"

    @property
    def rtsp_url(self) -> str:
        return self.rtsp_sources.get(self.default_rtsp_source, "")

    @rtsp_url.setter
    def rtsp_url(self, value: str) -> None:
        self.rtsp_sources[self.default_rtsp_source] = value

    @property
    def car_model_path(self) -> Path:
        return self.models.car_model_path

    @car_model_path.setter
    def car_model_path(self, value: str | Path) -> None:
        self.models.car_model_path = Path(value)

    @property
    def plate_model_path(self) -> Path:
        return self.models.plate_model_path


    @property
    def ocr_det_path(self) -> Path:
        return self.models.ocr_det_path

    @property
    def ocr_rec_path(self) -> Path:
        return self.models.ocr_rec_path

    @property
    def ocr_char_dict_path(self) -> Path:
        return self.models.ocr_char_dict_path

    @property
    def output_video(self) -> Path | None:
        return self.output_video_path

    @output_video.setter
    def output_video(self, value: str | Path | None) -> None:
        self.output_video_path = Path(value) if value else None
    @plate_model_path.setter
    def plate_model_path(self, value: str | Path) -> None:
        self.models.plate_model_path = Path(value)

    @property
    def vehicle_conf_threshold(self) -> float:
        return self.thresholds.vehicle_conf_threshold

    @vehicle_conf_threshold.setter
    def vehicle_conf_threshold(self, value: float) -> None:
        self.thresholds.vehicle_conf_threshold = float(value)

    @property
    def plate_conf_threshold(self) -> float:
        return self.thresholds.plate_conf_threshold

    @plate_conf_threshold.setter
    def plate_conf_threshold(self, value: float) -> None:
        self.thresholds.plate_conf_threshold = float(value)

    @property
    def nms_threshold(self) -> float:
        return self.thresholds.nms_threshold

    @nms_threshold.setter
    def nms_threshold(self, value: float) -> None:
        self.thresholds.nms_threshold = float(value)

    @property
    def ocr_regex_patterns(self) -> list[str]:
        return self.ocr_regex.patterns

    @ocr_regex_patterns.setter
    def ocr_regex_patterns(self, value: list[str]) -> None:
        self.ocr_regex.patterns = value

    @property
    def ocr_min_vehicle_width(self) -> int:
        return self.thresholds.ocr_min_vehicle_width

    @property
    def ocr_min_vehicle_area(self) -> int:
        return self.thresholds.ocr_min_vehicle_area

    @property
    def runtime_profiles(self) -> dict[str, RuntimeProfile]:
        return {
            "cpu": RuntimeProfile(name="cpu", providers=["CPUExecutionProvider"]),
            "gpu": RuntimeProfile(
                name="gpu",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
        }


def get_config() -> AppConfig:
    return AppConfig()

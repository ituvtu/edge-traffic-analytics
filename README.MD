# Edge Car Tracking (MVP)

MVP-конвеєр для edge car-tracking:
- ONNX Runtime only
- два профілі продуктивності: CPU / GPU
- OCR для UA + Latin з regex-валідацією
- live-візуалізація + CSV-лог

## Архітектура

Пайплайн у `src/main.py`:
1. RTSP capture (`src/io/rtsp_capture.py`) з окремим thread, last-frame buffer, reconnect/backoff.
2. Детекція авто ONNX (`src/inference/onnx_yolo.py`).
3. Трекінг (`src/pipeline/vehicle_tracker.py`) зі стабільними track ID.
4. Визначення кольору (`src/analytics/color_detection.py`) + кеш по track ID.
5. Plate detector + OCR (`src/pipeline/plate_ocr.py`, `src/ocr/easyocr_engine.py`) з одноразовою фіксацією номера на track ID.
6. Overlay (`src/ui/overlay.py`) і live display.
7. CSV-лог (`timestamp, track_id, plate, color, confidence`).

## Структура

```
src/
  config.py
  app.py
  main.py
  io/rtsp_capture.py
  inference/onnx_yolo.py
  pipeline/vehicle_tracker.py
  pipeline/plate_ocr.py
  analytics/color_detection.py
  ocr/easyocr_engine.py
  ui/overlay.py
```

## Встановлення

```bash
python -m pip install -r requirements.txt
```

## Моделі та конвертація

Очікувані шляхи:
- `pt/yolo10nano_car.pt`
- `pt/yolo10nano_plate.pt`
- `onnx/yolo10nano_car.onnx`
- `onnx/yolo10nano_plate.onnx`

```bash
python convert_to_onnx.py --help
python convert_to_onnx.py --dry-run
```

Dry-run перевіряє завантаження ONNX у:
- `CPUExecutionProvider`
- `CUDAExecutionProvider` (якщо доступний)

## Запуск

### CPU profile

```bash
python -m src.main --profile cpu --rtsp "rtsp://user:pass@host:554/stream"
```

### GPU profile

```bash
python -m src.main --profile gpu --rtsp "rtsp://user:pass@host:554/stream"
```

### Через PowerShell-скрипт

```powershell
./scripts/run.ps1 -Profile cpu -RtspUrl "rtsp://user:pass@host:554/stream"
./scripts/run.ps1 -Profile gpu -RtspUrl "rtsp://user:pass@host:554/stream"
```

## Конфігурація

Централізована в `src/config.py`:
- `RTSP_URL` (fallback для одного потоку)
- `RTSP_URLS` (формат: `main=rtsp://...;gate=rtsp://...`)
- `DEFAULT_RTSP_SOURCE`
- `CAR_MODEL_PATH`, `PLATE_MODEL_PATH`
- `VEHICLE_CONF_THRESHOLD`, `PLATE_CONF_THRESHOLD`, `NMS_THRESHOLD`
- `FRAME_SKIP`
- `OCR_REGEX_PATTERNS` (формат: `regex1;regex2;...`)
- ONNX providers через `PROFILE=cpu|gpu`

## Verification checklist

1. Встановлення:
   - `python -m pip install -r requirements.txt`
2. Перевірка моделей:
   - `python convert_to_onnx.py --help`
   - `python convert_to_onnx.py --dry-run`
3. Smoke test:
   - запуск RTSP 10+ хв без росту latency/буфера
4. Функціонально:
   - стабільні track ID
   - колір не “стрибає”
   - OCR виконується один раз на ID
   - CSV містить `timestamp, ID, plate, color, confidence`
5. Візуально:
   - live-вікно без помітних фризів

## Troubleshooting

- Якщо GPU профіль не стартує, перевірте сумісність `onnxruntime-gpu` + CUDA/cuDNN.
- Якщо OCR повільний, зменште `FRAME_SKIP` або підвищте пороги детекції.
- Якщо є дропи RTSP, перевірте мережу/кодек і використовуйте стабільний URI потік.

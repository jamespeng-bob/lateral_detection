"""Quick spike: what does localize() / classify() actually return?

Prints raw fields so we can decide if we have a class label in the
localize response, or whether we always need to call classify too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbols.call_symbol_localizer import (
    DEFAULT_CLASSIFICATION_ENDPOINT_ID,
    DEFAULT_GCS_BUCKET,
    DEFAULT_LOCALIZATION_ENDPOINT_ID,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    IsolatedSymbolClient,
    _resolve_credentials_file,
)


def main() -> int:
    client = IsolatedSymbolClient(
        credentials_file=_resolve_credentials_file(""),
        project_id=DEFAULT_PROJECT_ID,
        location=DEFAULT_LOCATION,
        localization_endpoint_id=DEFAULT_LOCALIZATION_ENDPOINT_ID,
        classification_endpoint_id=DEFAULT_CLASSIFICATION_ENDPOINT_ID,
        gcs_bucket=DEFAULT_GCS_BUCKET,
        nms_iou_thresh=0.5,
        conf_thresh=0.3,
    )

    train_dir = (ROOT.parent / "datasets" / "keypoints.v56-lat_only_baseline.coco" / "train").resolve()
    imgs = sorted(train_dir.glob("*.jpg"))
    img = imgs[0]
    print(f"=== localize() on {img.name} ===")
    dets = client.localize(str(img))
    print(f"  got {len(dets)} detections")
    if dets:
        print(f"  first detection keys: {sorted(dets[0].keys())}")
        print(f"  first 3 detections (full):")
        for i, d in enumerate(dets[:3]):
            print(f"    [{i}] {json.dumps(d, default=str, indent=2)}")

    print()
    print("=== classify() on first 5 detected crops ===")
    crop_ids = []
    for d in dets[:5]:
        for key in ("id", "ID", "crop_id"):
            value = d.get(key)
            if value:
                crop_ids.append(str(value))
                break
    print(f"  crop_ids passed: {crop_ids}")
    if crop_ids:
        cls = client.classify(crop_ids)
        print(f"  got {len(cls)} classification results")
        if cls:
            first = cls[0]
            if isinstance(first, dict):
                print(f"  first result keys: {sorted(first.keys())}")
            else:
                print(f"  first result type: {type(first).__name__}")
            print(f"  first 3 results (full):")
            for i, c in enumerate(cls[:3]):
                if isinstance(c, dict):
                    blob = json.dumps(c, default=str, indent=2)
                    print(f"    [{i}] {blob[:600]}")
                else:
                    print(f"    [{i}] (type={type(c).__name__}) {str(c)[:300]}")
    else:
        print("  no crop_ids available in detections - cannot test classify()")

    client.cleanup_uploads()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

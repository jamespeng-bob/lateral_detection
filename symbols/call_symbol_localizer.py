"""Isolated caller for symbol localization and classification endpoints.

This script does not import ``symbol_localizer.py`` or the
``symbol_validation_agent`` package. It contains the small amount of Vertex AI
and GCS plumbing needed to call the two endpoints directly:

    1. localization endpoint: ``parameters={"img_url": image_url}``
    2. classification endpoint: ``parameters={"crop_ids": "id1,id2,..."}``

Example:
    python -m symbols.call_symbol_localizer \
        --image datasets/keypoints.v56-lat_only_baseline.coco/test/<some>.jpg \
        --out-dir results/symbols/test_0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[reportMissingImports]
from google.cloud import aiplatform, storage
from google.oauth2 import service_account  # type: ignore[reportMissingImports]


# Default credentials path: <repo>/credentials/inference-428300-7af7f5da75dc.json.
# This module lives at <repo>/symbols/call_symbol_localizer.py, so
# ``parent.parent`` gives the repository root.
DEFAULT_CREDENTIALS_FILE = str(
    Path(__file__).resolve().parent.parent
    / 'credentials'
    / 'inference-428300-7af7f5da75dc.json'
)
DEFAULT_PROJECT_ID = '147252542820'
DEFAULT_LOCATION = 'us-west1'
DEFAULT_LOCALIZATION_ENDPOINT_ID = (
    f'projects/{DEFAULT_PROJECT_ID}/locations/{DEFAULT_LOCATION}/'
    'endpoints/1648990364733800448'
)
DEFAULT_CLASSIFICATION_ENDPOINT_ID = (
    f'projects/{DEFAULT_PROJECT_ID}/locations/{DEFAULT_LOCATION}/'
    'endpoints/9163730128117694464'
)
DEFAULT_GCS_BUCKET = 'public-bobyard'
GCS_UPLOAD_PREFIX = 'tmp_symbol_localizer'
MAX_RETRIES = 5


def _resolve_credentials_file(explicit: str) -> str:
    candidates: list[tuple[str, str]] = []
    if explicit:
        candidates.append(('argument', explicit))
    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        candidates.append((
            'GOOGLE_APPLICATION_CREDENTIALS',
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
        ))
    candidates.append(('default', DEFAULT_CREDENTIALS_FILE))

    for source, path in candidates:
        if Path(path).expanduser().is_file():
            resolved = str(Path(path).expanduser().resolve())
            logging.info('Using credentials from %s: %s', source, resolved)
            return resolved

    tried = '\n  '.join(f'{source}: {path}' for source, path in candidates)
    raise FileNotFoundError(
        'No usable Vertex credentials file found. Tried:\n  ' + tried
    )


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    xa = max(float(a['x1']), float(b['x1']))
    ya = max(float(a['y1']), float(b['y1']))
    xb = min(float(a['x2']), float(b['x2']))
    yb = min(float(a['y2']), float(b['y2']))
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (float(a['x2']) - float(a['x1'])) * (float(a['y2']) - float(a['y1']))
    area_b = (float(b['x2']) - float(b['x1'])) * (float(b['y2']) - float(b['y1']))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _nms(
    detections: list[dict[str, Any]],
    iou_thresh: float,
    conf_thresh: float,
) -> list[dict[str, Any]]:
    dets = [d for d in detections if float(d.get('conf', 0.0) or 0.0) >= conf_thresh]
    dets.sort(key=lambda d: float(d.get('conf', 0.0) or 0.0), reverse=True)

    keep: list[dict[str, Any]] = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_thresh]
    return keep


class IsolatedSymbolClient:
    """Direct Vertex/GCS client for localization and classification endpoints."""

    def __init__(
        self,
        *,
        credentials_file: str,
        project_id: str,
        location: str,
        localization_endpoint_id: str,
        classification_endpoint_id: str,
        gcs_bucket: str,
        nms_iou_thresh: float,
        conf_thresh: float,
    ) -> None:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file
        )
        aiplatform.init(
            project=project_id,
            location=location,
            credentials=credentials,
        )
        self._loc_endpoint = aiplatform.Endpoint(
            endpoint_name=localization_endpoint_id,
        )
        self._cls_endpoint = aiplatform.Endpoint(
            endpoint_name=classification_endpoint_id,
        )
        self._gcs_client = storage.Client(credentials=credentials)
        self._gcs_bucket_name = gcs_bucket
        self._nms_iou_thresh = nms_iou_thresh
        self._conf_thresh = conf_thresh
        self._uploaded_blobs: list[Any] = []
        self._upload_cache: dict[str, str] = {}

    def cleanup_uploads(self) -> None:
        for blob in self._uploaded_blobs:
            try:
                blob.delete()
            except Exception:  # noqa: BLE001
                logging.warning('Failed to delete temporary GCS blob', exc_info=True)
        self._uploaded_blobs.clear()
        self._upload_cache.clear()

    def _upload_to_gcs(self, local_path: str) -> str:
        resolved = str(Path(local_path).expanduser().resolve())
        if resolved in self._upload_cache:
            return self._upload_cache[resolved]

        path = Path(resolved)
        blob_name = f'{GCS_UPLOAD_PREFIX}/{uuid.uuid4().hex}_{path.name}'
        bucket = self._gcs_client.bucket(self._gcs_bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(path))
        self._uploaded_blobs.append(blob)

        encoded_name = urllib.parse.quote(blob_name, safe='/')
        url = f'https://storage.googleapis.com/{self._gcs_bucket_name}/{encoded_name}'
        self._upload_cache[resolved] = url
        logging.info('Uploaded %s to %s', path.name, url)
        return url

    def _resolve_image_url(self, image: str) -> str:
        if image.startswith(('http://', 'https://', 'gs://')):
            return image
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Image not found: {path}')
        return self._upload_to_gcs(str(path))

    def _predict_with_retry(
        self,
        endpoint: Any,
        *,
        instances: list[Any],
        parameters: dict[str, Any],
    ) -> list[Any]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = endpoint.predict(
                    instances=instances,
                    parameters=parameters,
                )
                return list(result.predictions or [])
            except Exception as exc:
                message = str(exc)
                retryable = (
                    '503' in message
                    or '409' in message
                    or 'Already running' in message
                )
                if not retryable or attempt == MAX_RETRIES:
                    raise
                base = 2 if ('409' in message or 'Already running' in message) else 1
                wait_s = min(base * (2 ** attempt), 60)
                logging.warning(
                    'Endpoint call failed on attempt %d/%d; retrying in %ss: %s',
                    attempt,
                    MAX_RETRIES,
                    wait_s,
                    message,
                )
                time.sleep(wait_s)
        return []

    def localize(self, image: str) -> list[dict[str, Any]]:
        resolved_url = self._resolve_image_url(image)
        raw = self._predict_with_retry(
            self._loc_endpoint,
            instances=[0],
            parameters={'img_url': resolved_url},
        )
        raw_dets = [d for d in raw if isinstance(d, dict)]
        filtered = _nms(raw_dets, self._nms_iou_thresh, self._conf_thresh)
        logging.info(
            'NMS kept %d/%d detections (iou=%s, conf>=%s)',
            len(filtered),
            len(raw_dets),
            self._nms_iou_thresh,
            self._conf_thresh,
        )
        return filtered

    def classify(self, crop_ids: list[str]) -> list[Any]:
        if not crop_ids:
            return []
        return self._predict_with_retry(
            self._cls_endpoint,
            instances=[0],
            parameters={'crop_ids': ','.join(crop_ids)},
        )


def _make_client(args: argparse.Namespace) -> IsolatedSymbolClient:
    credentials_file = _resolve_credentials_file(args.credentials_file)
    return IsolatedSymbolClient(
        credentials_file=credentials_file,
        project_id=args.project_id,
        location=args.location,
        localization_endpoint_id=args.localization_endpoint_id,
        classification_endpoint_id=args.classification_endpoint_id,
        gcs_bucket=args.gcs_bucket,
        nms_iou_thresh=args.nms_iou_thresh,
        conf_thresh=args.conf_thresh,
    )


def _crop_id(record: dict[str, Any]) -> str | None:
    for key in ('id', 'ID', 'crop_id'):
        value = record.get(key)
        if value is not None and value != '':
            return str(value)
    return None


def _normalize_detections(raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    detections: list[dict[str, Any]] = []
    crop_ids: list[str] = []

    for record in raw or []:
        crop_id = _crop_id(record)
        if crop_id is None:
            continue
        try:
            bbox = [
                float(record['x1']),
                float(record['y1']),
                float(record['x2']),
                float(record['y2']),
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        detections.append({
            'crop_id': crop_id,
            'bbox_xyxy': bbox,
            'confidence': float(record.get('conf', record.get('confidence', 1.0)) or 1.0),
            'class_id': int(record.get('class_id', record.get('class', 0)) or 0),
            'class_name': str(
                record.get('class_name')
                or record.get('label')
                or record.get('name')
                or 'symbol'
            ),
        })
        crop_ids.append(crop_id)

    return detections, crop_ids


def _classify_in_batches(
    client: IsolatedSymbolClient,
    crop_ids: list[str],
    batch_size: int,
    include_embeddings_json: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    classifications: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for start in range(0, len(crop_ids), batch_size):
        chunk = crop_ids[start:start + batch_size]
        result = client.classify(chunk) or []
        if len(result) != len(chunk):
            raise RuntimeError(
                f'classify returned {len(result)} rows for {len(chunk)} crop ids'
            )

        for crop_id, record in zip(chunk, result):
            if not isinstance(record, dict):
                record = {'raw': record}
            embedding = (
                record.get('embedding')
                or record.get('embeddings')
                or record.get('EMBEDDINGS')
            )
            if not embedding:
                raise RuntimeError(f'classify returned no embedding for crop_id={crop_id}')
            saved_record = dict(record)
            if not include_embeddings_json:
                saved_record.pop('embedding', None)
                saved_record.pop('embeddings', None)
                saved_record.pop('EMBEDDINGS', None)
                saved_record['embedding_dim'] = len(embedding)
            classifications.append({
                'crop_id': crop_id,
                'classification': saved_record,
            })
            embeddings.append([float(value) for value in embedding])

    if not embeddings:
        return classifications, np.empty((0, 0), dtype=np.float32)
    return classifications, np.asarray(embeddings, dtype=np.float32)


def _write_outputs(
    out_dir: Path,
    image: str,
    raw_localizations: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    embeddings: np.ndarray,
    include_embeddings_json: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        'image': image,
        'num_raw_localizations': len(raw_localizations),
        'num_detections': len(detections),
        'embedding_shape': list(embeddings.shape),
        'detections': detections,
        'classifications': classifications,
    }
    if include_embeddings_json:
        payload['embeddings'] = embeddings.astype(float).tolist()

    (out_dir / 'symbol_localizer_output.json').write_text(json.dumps(payload, indent=2))
    (out_dir / 'raw_localizations.json').write_text(json.dumps(raw_localizations, indent=2))
    if embeddings.size:
        np.save(out_dir / 'embeddings.npy', embeddings.astype(np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Isolated symbol localizer + classification endpoint caller.'
    )
    parser.add_argument('--image', required=True, help='Local image path or public image URL.')
    parser.add_argument('--out-dir', required=True, help='Output directory.')
    parser.add_argument('--credentials-file', default='',
                        help='Service account JSON. Defaults to GOOGLE_APPLICATION_CREDENTIALS or the built-in workspace path.')
    parser.add_argument('--project-id', default=DEFAULT_PROJECT_ID)
    parser.add_argument('--location', default=DEFAULT_LOCATION)
    parser.add_argument('--localization-endpoint-id',
                        default=DEFAULT_LOCALIZATION_ENDPOINT_ID)
    parser.add_argument('--classification-endpoint-id',
                        default=DEFAULT_CLASSIFICATION_ENDPOINT_ID)
    parser.add_argument('--gcs-bucket', default=DEFAULT_GCS_BUCKET)
    parser.add_argument('--nms-iou-thresh', type=float, default=0.5)
    parser.add_argument('--conf-thresh', type=float, default=0.0)
    parser.add_argument('--classify-batch-size', type=int, default=64)
    parser.add_argument('--include-embeddings-json', action='store_true',
                        help='Inline embeddings in JSON in addition to embeddings.npy.')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    image = args.image
    if not image.startswith(('http://', 'https://', 'gs://')):
        image = str(Path(image).expanduser().resolve())

    client = _make_client(args)
    try:
        logging.info('Calling localize on %s', image)
        raw_localizations = client.localize(image) or []
        detections, crop_ids = _normalize_detections(raw_localizations)
        logging.info('Localize returned %d usable detections', len(detections))

        classifications, embeddings = _classify_in_batches(
            client,
            crop_ids,
            batch_size=args.classify_batch_size,
            include_embeddings_json=args.include_embeddings_json,
        )
        logging.info('Classify returned embedding shape %s', tuple(embeddings.shape))

        out_dir = Path(args.out_dir).expanduser().resolve()
        _write_outputs(
            out_dir=out_dir,
            image=image,
            raw_localizations=raw_localizations,
            detections=detections,
            classifications=classifications,
            embeddings=embeddings,
            include_embeddings_json=args.include_embeddings_json,
        )
    finally:
        client.cleanup_uploads()

    print(f'Detections: {len(detections)}')
    print(f'Embeddings: {tuple(embeddings.shape)}')
    print(f'Wrote: {out_dir / "symbol_localizer_output.json"}')
    print(f'Wrote: {out_dir / "raw_localizations.json"}')
    if embeddings.size:
        print(f'Wrote: {out_dir / "embeddings.npy"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

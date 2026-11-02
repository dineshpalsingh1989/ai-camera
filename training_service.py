"""Local reviewed datasets, background training jobs, model registry and Roboflow transfers."""
import copy
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from urllib.parse import urlparse, quote
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
import numpy as np
import requests
from flask import Blueprint, jsonify, request, send_file
from werkzeug.exceptions import HTTPException


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def bounded(value, low, high, kind=int):
    number = kind(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f'Value must be between {low} and {high}')
    return number


def slug(value):
    value = str(value).strip()
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,120}', value):
        raise ValueError('Use a valid workspace, project or model identifier')
    return value


def safe_extract(archive, destination):
    """Extract data, never scripts, symlinks or paths outside the new dataset directory."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) > 100000 or sum(i.file_size for i in entries) > 4 * 1024**3:
            raise ValueError('Dataset exceeds 100,000 files or 4 GB unpacked')
        for entry in entries:
            name = entry.filename.replace('\\', '/')
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or ':' in name or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe path in dataset archive')
            if entry.is_dir():
                continue
            if target.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.txt', '.yaml', '.yml', '.json'}:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)


def prepare_dataset(folder):
    """Normalize YOLO detection exports without honoring download commands or external paths."""
    import yaml
    folder = Path(folder).resolve()
    candidates = list(folder.rglob('data.yaml')) + list(folder.rglob('data.yml'))
    if len(candidates) != 1:
        raise ValueError('ZIP must contain one YOLO detection data.yaml')
    original = candidates[0]
    data = yaml.safe_load(original.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Invalid dataset configuration')
    names = data.get('names')
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if not isinstance(names, list) or not names or len(names) > 200 or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError('Dataset needs 1–200 named classes')
    if len(set(names)) != len(names):
        raise ValueError('Class names must be unique')
    config = {'path': str(original.parent), 'names': names, 'nc': len(names)}
    counts, hashes = {}, {}
    for split, aliases in [('train', ['train']), ('val', ['valid', 'val']), ('test', ['test'])]:
        options = [original.parent / alias / 'images' for alias in aliases]
        options += [original.parent / 'images' / alias for alias in aliases]
        raw = data.get(split)
        if isinstance(raw, str):
            possible = (original.parent / raw).resolve()
            if possible.is_relative_to(folder):
                options.append(possible)
        images_dir = next((p for p in options if p.is_dir() and p.resolve().is_relative_to(folder)), None)
        if images_dir is None:
            if split == 'test':
                continue
            raise ValueError(f'Missing {split}/images directory')
        images = [p for p in images_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not images:
            raise ValueError(f'The {split} split has no images')
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError('Identical images appear in different splits; remove leakage before training')
            hashes[digest] = split
            parts = list(image.parts)
            parts[len(images_dir.parts) - 1 if images_dir.name == 'images' else len(images_dir.parts) - 2] = 'labels'
            label = Path(*parts).with_suffix('.txt')
            if not label.is_file():
                raise ValueError('Every image needs a label file (empty for reviewed background images)')
            for row in label.read_text(encoding='utf-8-sig').splitlines():
                values = row.split()
                if not values:
                    continue
                if len(values) != 5:
                    raise ValueError('Only YOLO object-detection boxes are supported, not segmentation or pose')
                idx = int(values[0]); xywh = [float(v) for v in values[1:]]
                if not 0 <= idx < len(names) or not all(math.isfinite(v) and 0 <= v <= 1 for v in xywh) or min(xywh[2:]) <= 0:
                    raise ValueError('Invalid YOLO label coordinates or class index')
        config[split] = str(images_dir.resolve())
        counts[split] = len(images)
    original.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return original, names, counts


class TrainingService:
    def __init__(self, context, root=None):
        self.ctx = context
        self.root = Path(root or Path(context['BASE']) / 'ai-workspace').resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self.cancelled = set()
        self.last_sample = {}
        self.last_signature = {}
        self.state_file = self.root / 'state.json'
        self.connection_file = self.root / 'connection.json'
        self.state = {'photos': {}, 'datasets': {}, 'models': {}, 'jobs': {}, 'history': [], 'auto_collect': {}}
        if self.state_file.exists():
            self.state.update(json.loads(self.state_file.read_text(encoding='utf-8')))
        for job in self.state['jobs'].values():
            if job['status'] in {'queued', 'running', 'cancelling'}:
                job.update(status='interrupted', message='Server restarted; start a new job to retry')
        self.connection = {'api_key': '', 'workspace': '', 'project': ''}
        if self.connection_file.exists():
            self.connection.update(json.loads(self.connection_file.read_text(encoding='utf-8')))
        self.save()

    def save(self):
        with self.lock:
            atomic_json(self.state_file, self.state)

    def public(self):
        with self.lock:
            snapshot = copy.deepcopy(self.state)
        snapshot['photos'] = sorted(snapshot['photos'].values(), key=lambda p: p['created'], reverse=True)
        for key in ('datasets', 'models', 'jobs'):
            snapshot[key] = sorted(snapshot[key].values(), key=lambda p: p['created'], reverse=True)
        snapshot['connection'] = {k: v for k, v in self.connection.items() if k != 'api_key'}
        snapshot['connection']['key_set'] = bool(self.connection['api_key'])
        snapshot['active_model'] = self.ctx['settings']['custom_model']
        snapshot['can_rollback'] = bool(snapshot.pop('history'))
        snapshot['engine'] = 'Local YOLO training; CPU by default, CUDA when selected and available'
        return snapshot

    def item(self, category, key):
        item = self.state[category].get(key)
        if not item:
            raise ValueError(f'{category[:-1].capitalize()} not found')
        return item

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Invalid workspace path')
        return path

    def add_photo(self, jpg, camera='upload', source='Uploaded image', split='train', suggestions=None):
        if split not in {'train', 'val', 'test'}:
            raise ValueError('Choose train, val or test split')
        if len(jpg) > 20 * 1024**2:
            raise ValueError('Images must be smaller than 20 MB')
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size > 60_000_000:
            raise ValueError('Use a JPEG or PNG image up to 20 megapixels')
        jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        digest = hashlib.sha256(jpg).hexdigest()
        with self.lock:
            existing = next((p for p in self.state['photos'].values() if p['hash'] == digest), None)
            if existing:
                return copy.deepcopy(existing)
            if len(self.state['photos']) >= 5000:
                raise ValueError('Photo library is full (5,000). Remove unused photos before collecting more')
            pid = uuid.uuid4().hex[:16]
            target = self.path(f'photos/{pid}.jpg')
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(jpg)
            photo = {'id': pid, 'file': f'photos/{pid}.jpg', 'camera': camera, 'source': source,
                     'created': time.time(), 'width': frame.shape[1], 'height': frame.shape[0], 'hash': digest,
                     'reviewed': False, 'boxes': suggestions or [], 'split': split, 'uploads': {}}
            self.state['photos'][pid] = photo
            self.save()
            return copy.deepcopy(photo)

    def sample(self, cid, frame, detections):
        policy = self.state['auto_collect'].get(cid, {})
        if not policy.get('enabled') or frame is None or time.time() - self.last_sample.get(cid, 0) < policy['interval']:
            return
        uncertain = [d for d in detections if d[1] <= policy['confidence']]
        if not uncertain:
            return
        self.last_sample[cid] = time.time()
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(float)
        if cid in self.last_signature and np.abs(thumb - self.last_signature[cid]).mean() < 5:
            return
        self.last_signature[cid] = thumb
        h, w = frame.shape[:2]
        boxes = [{'label': d[0], 'x1': max(0, d[2]/w), 'y1': max(0, d[3]/h),
                  'x2': min(1, d[4]/w), 'y2': min(1, d[5]/h)} for d in detections]
        self.add_photo(cv2.imencode('.jpg', frame)[1].tobytes(), cid, 'Uncertain detection · needs review', policy['split'], boxes)

    def update_job(self, jid, **values):
        with self.lock:
            self.state['jobs'][jid].update(values, updated=time.time())
            self.save()

    def start(self, kind, work):
        with self.lock:
            if any(j['status'] in {'queued', 'running', 'cancelling'} for j in self.state['jobs'].values()):
                raise ValueError('Another job is running. Wait for it to finish or cancel it first')
            jid = uuid.uuid4().hex[:12]
            self.state['jobs'][jid] = {'id': jid, 'kind': kind, 'created': time.time(), 'status': 'queued', 'progress': 0, 'message': 'Queued'}
            done = sorted((j for j in self.state['jobs'].values() if j['status'] in ('completed', 'failed', 'cancelled')),
                          key=lambda j: j['created'])
            for old in done[:-20]:  # keep the job log bounded to the 20 most recent finished jobs
                self.state['jobs'].pop(old['id'], None)
            self.save()
        def run():
            try:
                self.check_cancel(jid)
                self.update_job(jid, status='running', message='Starting')
                result = work(jid) or {}
                self.check_cancel(jid)
                self.update_job(jid, status='completed', progress=100, message='Completed', result=result)
            except Exception as e:
                message = self.redact(str(e))
                self.update_job(jid, status='cancelled' if jid in self.cancelled else 'failed', message=message[:700])
            finally:
                self.processes.pop(jid, None)
        threading.Thread(target=run, daemon=True).start()
        return {'job_id': jid}

    def clear_jobs(self):
        """Remove finished jobs from the log; running/queued jobs are kept."""
        with self.lock:
            self.state['jobs'] = {jid: j for jid, j in self.state['jobs'].items()
                                  if j['status'] in ('queued', 'running', 'cancelling')}
            self.save()
        return {'ok': True}

    def check_cancel(self, jid):
        if jid in self.cancelled:
            raise ValueError('Cancelled by user')

    def redact(self, text):
        if self.connection['api_key']:
            text = text.replace(self.connection['api_key'], '[redacted]')
        return re.sub(r'https?://\S+', '[remote service]', text)

    def auto_split(self):
        """Assign ~20% of reviewed photos to validation and the rest to training, spread evenly across
        capture order so the split is reproducible and validation is not all from one moment."""
        with self.lock:
            reviewed = sorted((p for p in self.state['photos'].values() if p['reviewed']), key=lambda p: p['created'])
            n = len(reviewed)
            if n < 4:
                raise ValueError('Need at least 4 reviewed photos to auto-split (2 training and 2 validation).')
            n_val = min(max(2, round(n * 0.2)), n - 2)  # keep at least 2 in each split
            val_idx = {int(j * n / n_val) for j in range(n_val)}
            for i, photo in enumerate(reviewed):
                photo['split'] = 'val' if i in val_idx else 'train'
            self.save()
        return {'train': n - len(val_idx), 'val': len(val_idx)}

    def build_dataset(self, selected=None):
        with self.lock:
            photos = copy.deepcopy([p for p in self.state['photos'].values() if p['reviewed'] and (selected is None or p['id'] in selected)])
        names = sorted({b['label'] for p in photos for b in p['boxes']})
        counts = {s: sum(p['split'] == s for p in photos) for s in ('train', 'val', 'test')}
        if counts['train'] < 2 or counts['val'] < 2 or not names:
            detail = f"you have {counts['train']} training and {counts['val']} validation reviewed"
            if not names:
                detail += " but no labeled objects"
            raise ValueError(f"Need at least 2 training and 2 validation reviewed images with labeled objects — {detail}. Capture more and set some photos to the Validation split.")
        did = uuid.uuid4().hex[:12]
        folder = self.path('datasets/' + did)
        folder.mkdir(parents=True)
        for photo in photos:
            target = folder / photo['split']
            (target / 'images').mkdir(parents=True, exist_ok=True)
            (target / 'labels').mkdir(exist_ok=True)
            shutil.copyfile(self.path(photo['file']), target / 'images' / (photo['id'] + '.jpg'))
            rows = []
            for b in photo['boxes']:
                rows.append(f"{names.index(b['label'])} {(b['x1']+b['x2'])/2:.7f} {(b['y1']+b['y2'])/2:.7f} {b['x2']-b['x1']:.7f} {b['y2']-b['y1']:.7f}")
            (target / 'labels' / (photo['id'] + '.txt')).write_text('\n'.join(rows), encoding='utf-8')
        import yaml
        (folder / 'data.yaml').write_text(yaml.safe_dump({'names': names, 'train': 'train/images', 'val': 'val/images', **({'test': 'test/images'} if counts['test'] else {})}), encoding='utf-8')
        return self.register_dataset(did, folder, 'Reviewed camera images')

    def register_dataset(self, did, folder, name):
        config, names, counts = prepare_dataset(folder)
        data = {'id': did, 'name': name[:120], 'created': time.time(), 'yaml': str(config.relative_to(self.root)), 'classes': names, 'counts': counts}
        with self.lock:
            self.state['datasets'][did] = data
            self.save()
        return data

    def worker(self, jid, payload):
        folder = self.path('jobs/' + jid)
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'request.json', payload)
        command = [sys.executable, str(Path(__file__).with_name('training_worker.py')), str(folder)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (folder / 'worker.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, creationflags=flags)
            self.processes[jid] = process
            while process.poll() is None:
                if jid in self.cancelled:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    self.check_cancel(jid)
                try:
                    progress = json.loads((folder / 'progress.json').read_text(encoding='utf-8'))
                    self.update_job(jid, progress=progress.get('progress', 0), message=progress.get('message', 'Working'))
                except (OSError, ValueError):
                    pass
                time.sleep(.5)
        result_file = folder / 'result.json'
        result = json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else {}
        if process.returncode or result.get('error'):
            raise ValueError(result.get('error', 'Worker stopped unexpectedly; see the local job log'))
        return result

    def register_model(self, jid, path, name, dataset=None, classes=None):
        inspected = self.worker(jid, {'action': 'inspect', 'model': str(path), 'classes': classes})
        self.check_cancel(jid)
        mid = uuid.uuid4().hex[:12]
        target = self.path('models/' + mid + '.pt')
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(inspected.get('model', path), target)
        model = {'id': mid, 'name': name[:120], 'created': time.time(), 'file': str(target.relative_to(self.root)), 'classes': inspected['classes'], 'evaluations': {}, 'dataset': dataset}
        with self.lock:
            self.state['models'][mid] = model
            self.save()
        return model

    def train(self, jid, data):
        dataset = copy.deepcopy(self.item('datasets', data['dataset']))
        base = Path(self.ctx['BASE']) / 'yolov8n.pt'
        if data.get('base_model'):
            base = self.path(self.item('models', data['base_model'])['file'])
        if not base.is_file():
            raise ValueError('Place yolov8n.pt in the app directory or import a compatible base model')
        result = self.worker(jid, {'action': 'train', 'model': str(base), 'data': str(self.path(dataset['yaml'])),
                                  'epochs': data['epochs'], 'imgsz': data['imgsz'], 'batch': data['batch'], 'device': data['device']})
        self.check_cancel(jid)
        mid = uuid.uuid4().hex[:12]
        target = self.path('models/' + mid + '.pt'); target.parent.mkdir(exist_ok=True)
        shutil.copyfile(result['model'], target)
        model = {'id': mid, 'name': data.get('name') or 'Local training ' + time.strftime('%d %b %H:%M'),
                 'created': time.time(), 'file': str(target.relative_to(self.root)), 'classes': result['classes'],
                 'dataset': dataset['id'], 'evaluations': {dataset['id']: result['metrics']}}
        with self.lock:
            self.state['models'][mid] = model
            self.save()
        return {'model_id': mid, 'metrics': result['metrics']}

    def evaluate(self, jid, mid, did):
        model, dataset = copy.deepcopy(self.item('models', mid)), copy.deepcopy(self.item('datasets', did))
        if model['classes'] != dataset['classes']:
            raise ValueError('Dataset class names and order must match the model')
        result = self.worker(jid, {'action': 'validate', 'model': str(self.path(model['file'])), 'data': str(self.path(dataset['yaml']))})
        self.check_cancel(jid)
        with self.lock:
            self.state['models'][mid]['evaluations'][did] = result['metrics']
            self.save()
        return result

    def activate(self, mid, camera_ids):
        model = self.item('models', mid)
        if not model['evaluations']:
            raise ValueError('Evaluate this model on a validation dataset before activation')
        if not camera_ids or any(cid not in {c['id'] for c in self.ctx['settings']['cameras']} for cid in camera_ids):
            raise ValueError('Choose at least one configured camera')
        # Load the candidate before touching the live model, so an incompatible import cannot replace it.
        path = str(self.path(model['file']))
        candidate = self.ctx['CustomDetector'](path, model['classes'])
        with self.lock, self.ctx['custom_lock']:
            settings = self.ctx['settings']
            previous = {'custom_model': settings['custom_model'], 'custom_classes': settings['custom_classes'][:],
                        'cameras': {c['id']: c['features']['custom'] for c in settings['cameras']}}
            updated = copy.deepcopy(settings)
            updated.update(custom_model=path, custom_classes=model['classes'])
            for camera in updated['cameras']:
                camera['features']['custom'] = camera['id'] in camera_ids
            self.ctx['save_settings'](updated)
            self.ctx['settings'] = updated
            self.ctx['custom_det'], self.ctx['custom_error'] = candidate, ''
            self.state['history'].append(previous)
            self.state['history'] = self.state['history'][-10:]
            self.save()
        self.ctx['sync_cameras']()
        return {'ok': True}

    def rollback(self):
        with self.lock:
            if not self.state['history']:
                raise ValueError('No previous activation to restore')
            previous = self.state['history'][-1]
            candidate = self.ctx['CustomDetector'](previous['custom_model'], previous['custom_classes'])
            with self.ctx['custom_lock']:
                updated = copy.deepcopy(self.ctx['settings'])
                updated.update({k: previous[k] for k in ('custom_model', 'custom_classes')})
                for camera in updated['cameras']:
                    camera['features']['custom'] = previous['cameras'].get(camera['id'], False)
                self.ctx['save_settings'](updated)
                self.ctx['settings'] = updated
                self.ctx['custom_det'], self.ctx['custom_error'] = candidate, ''
                self.state['history'].pop(); self.save()
        self.ctx['sync_cameras']()

    def delete_model(self, mid):
        model = self.item('models', mid)
        path = str(self.path(model['file']))
        if path == str(self.ctx['settings'].get('custom_model') or ''):
            raise ValueError('This model is active. Activate another model or roll back before deleting it.')
        with self.lock:
            self.state['models'].pop(mid, None)
            # drop any rollback points that restore this model, so it isn't pinned undeletable forever
            self.state['history'] = [h for h in self.state['history'] if h.get('custom_model') != path]
            self.save()
        self.path(model['file']).unlink(missing_ok=True)
        return {'ok': True}

    def delete_dataset(self, did):
        self.item('datasets', did)  # 404 if unknown
        with self.lock:
            self.state['datasets'].pop(did, None)
            self.save()
        shutil.rmtree(self.path('datasets/' + did), ignore_errors=True)  # snapshot is immutable; safe to drop
        for z in (self.path('downloads/' + did + '.zip'), self.path('downloads/' + did + '-export.zip')):
            z.unlink(missing_ok=True)
        return {'ok': True}
        return {'ok': True}

    def rf(self, suffix='', method='GET', params=None, **kwargs):
        connection = self.connection.copy()
        if not all(connection[k] for k in ('api_key', 'workspace', 'project')):
            raise ValueError('Save your Roboflow API key, workspace and project first')
        path = suffix if suffix.startswith('/dataset/') else f"/{connection['workspace']}/{connection['project']}" + suffix
        try:
            response = requests.request(method, 'https://api.roboflow.com' + path,
                                        params={'api_key': connection['api_key'], **(params or {})}, timeout=(10, 90), **kwargs)
            if not response.ok:
                raise ValueError(f'Roboflow returned HTTP {response.status_code}. Check access, project/version, export support and account credits')
            data = response.json()
            if data.get('error'):
                raise ValueError('Roboflow rejected the request. Check the selected project and version')
            return data
        except requests.RequestException:
            if method == 'POST' and suffix.endswith('/v2/trainings'):
                raise ValueError('Cloud training submission could not be confirmed. Check cloud status or your Roboflow project before submitting again') from None
            raise ValueError('Cannot reach Roboflow. Check your internet connection and retry') from None

    def download(self, jid, url, target, limit=1024**3):
        # Follow only public HTTPS artifact URLs returned by Roboflow, never local network destinations.
        for _ in range(6):
            parsed = urlparse(url)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.port not in (None, 443):
                raise ValueError('Invalid artifact download URL')
            addresses = socket.getaddrinfo(parsed.hostname, 443)
            if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
                raise ValueError('Artifact download must use a public host')
            with requests.get(url, stream=True, allow_redirects=False, timeout=(10, 60)) as response:
                if response.is_redirect:
                    from urllib.parse import urljoin
                    url = urljoin(url, response.headers['Location']); continue
                if not response.ok:
                    raise ValueError(f'Artifact download failed (HTTP {response.status_code})')
                size = 0
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('wb') as f:
                    for chunk in response.iter_content(1024*1024):
                        self.check_cancel(jid)
                        size += len(chunk)
                        if size > limit:
                            raise ValueError('Download is larger than 1 GB')
                        f.write(chunk)
                        self.update_job(jid, message=f'Downloaded {size // 1024**2} MB')
                return
        raise ValueError('Too many download redirects')

    def rf_upload(self, jid, ids):
        project = self.connection['project']
        destination = self.connection['workspace'] + '/' + project
        total = len(ids)
        for index, pid in enumerate(ids):
            self.check_cancel(jid)
            photo = copy.deepcopy(self.item('photos', pid))
            fingerprint = hashlib.sha256(json.dumps([photo['boxes'], photo['reviewed'], photo['split']], sort_keys=True).encode()).hexdigest()
            if photo.get('uploads', {}).get(destination) == fingerprint:
                continue
            with self.path(photo['file']).open('rb') as f:
                result = self.rf(f'/dataset/{project}/upload', 'POST', params={'batch': 'ai-camera', 'tag': photo['camera']},
                                 data={'name': pid + '.jpg', 'split': 'valid' if photo['split'] == 'val' else photo['split']},
                                 files={'file': (pid + '.jpg', f, 'image/jpeg')})
            if not (result.get('success') or result.get('duplicate')) or not result.get('id'):
                raise ValueError('Roboflow did not confirm the image upload')
            if photo['reviewed']:
                annotation = Element('annotation')
                SubElement(annotation, 'filename').text = pid + '.jpg'
                size = SubElement(annotation, 'size')
                for key, value in [('width', photo['width']), ('height', photo['height']), ('depth', 3)]:
                    SubElement(size, key).text = str(value)
                for box in photo['boxes']:
                    obj = SubElement(annotation, 'object'); SubElement(obj, 'name').text = box['label']
                    bound = SubElement(obj, 'bndbox')
                    for name, key, scale in [('xmin','x1',photo['width']), ('ymin','y1',photo['height']), ('xmax','x2',photo['width']), ('ymax','y2',photo['height'])]:
                        SubElement(bound, name).text = str(round(box[key] * scale))
                reply = self.rf(f"/dataset/{project}/annotate/{quote(str(result['id']), safe='')}", 'POST',
                                params={'name': pid + '.xml', 'overwrite': 'true'},
                                json={'annotationFile': tostring(annotation, encoding='unicode')})
                if not reply.get('success'):
                    raise ValueError('Image uploaded, but Roboflow did not confirm its annotation. Retry after checking the project')
            with self.lock:
                self.state['photos'][pid]['uploads'][destination] = fingerprint
                self.save()
            self.update_job(jid, progress=round((index+1)/total*100), message=f'Uploaded {index+1} / {total} images')
        return {'images': total}

    def rf_dataset(self, jid, version):
        for _ in range(180):
            self.check_cancel(jid)
            result = self.rf(f'/{version}/yolov8')
            link = result.get('export', {}).get('link')
            if link:
                break
            self.update_job(jid, message='Roboflow is preparing the dataset export')
            time.sleep(2)
        else:
            raise ValueError('Export is still preparing. Retry later')
        did = uuid.uuid4().hex[:12]
        archive = self.path('downloads/' + did + '.zip')
        self.download(jid, link, archive)
        folder = self.path('datasets/' + did)
        try:
            safe_extract(archive, folder)
            return self.register_dataset(did, folder, f"{self.connection['project']} v{version}")
        finally:
            archive.unlink(missing_ok=True)

    def rf_model(self, jid, model_id):
        response = self.rf('/' + quote(model_id, safe='') + '/ptFile')
        if not response.get('weightsUrl'):
            raise ValueError('This model has no downloadable PyTorch weights')
        target = self.path('downloads/' + jid + '.pt')
        try:
            self.download(jid, response['weightsUrl'], target)
            model = self.register_model(jid, target, self.connection['project'] + ' / ' + model_id)
            return {'model_id': model['id']}
        finally:
            target.unlink(missing_ok=True)


def register_training(app, context, root=None):
    service = TrainingService(context, root)
    bp = Blueprint('training', __name__, url_prefix='/api/training')

    @bp.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, (ValueError, TypeError, KeyError, zipfile.BadZipFile)):
            return jsonify(error=service.redact(str(exc))), 400
        return jsonify(error='Training operation failed. Check the server log and try again'), 500

    @bp.get('')
    def overview():
        return jsonify(service.public())

    @bp.get('/photos/<pid>/image')
    def image(pid):
        return send_file(service.path(service.item('photos', pid)['file']), mimetype='image/jpeg')

    @bp.post('/photos')
    def upload_photo():
        split = request.form.get('split', 'train')
        files = request.files.getlist('files')
        if not files or len(files) > 100:
            raise ValueError('Select between 1 and 100 images')
        return jsonify(photos=[service.add_photo(f.read(20*1024**2+1), source=f.filename[:100], split=split) for f in files])

    @bp.post('/capture')
    def capture():
        data = request.get_json()
        cam = context['cameras'].get(data.get('camera'))
        if not cam or cam.raw is None or not cam.online or not cam.last_frame or time.time() - cam.last_frame > 15:
            raise ValueError('The camera must be connected and have a current frame')
        frame = cam.raw
        h, w = frame.shape[:2]
        suggestions = [{'label': d[0], 'x1': max(0, d[2] / w), 'y1': max(0, d[3] / h),
                        'x2': min(1, d[4] / w), 'y2': min(1, d[5] / h)} for d in getattr(cam, 'last_dets', [])]
        return jsonify(service.add_photo(cv2.imencode('.jpg', frame)[1].tobytes(), cam.id, cam.name, data.get('split', 'train'), suggestions))

    @bp.post('/import-collected')
    def import_collected():
        data = request.get_json(); cid = slug(data.get('camera', ''))
        if cid not in {c['id'] for c in context['settings']['cameras']}:
            raise ValueError('Camera not found')
        files = sorted((Path(context['TRAIN_DIR']) / cid).glob('*.jpg'))[-500:]
        count = 0
        for file in files:
            service.add_photo(file.read_bytes(), cid, 'Collected camera photo', data.get('split', 'train')); count += 1
        return jsonify(imported=count)

    @bp.route('/photos/<pid>', methods=['PATCH', 'DELETE'])
    def review(pid):
        with service.lock:
            photo = service.item('photos', pid)
            if request.method == 'DELETE':
                if any(j['status'] in {'running', 'queued', 'cancelling'} for j in service.state['jobs'].values()):
                    raise ValueError('Wait for the current job before deleting photos')
                service.path(photo['file']).unlink(missing_ok=True)
                del service.state['photos'][pid]; service.save(); return jsonify(ok=True)
            data = request.get_json()
            split = data.get('split', photo['split'])
            if split not in {'train', 'val', 'test'}:
                raise ValueError('Invalid dataset split')
            boxes = data.get('boxes', photo['boxes'])
            if not isinstance(boxes, list) or len(boxes) > 200:
                raise ValueError('Use at most 200 boxes per image')
            cleaned = []
            for b in boxes:
                label = str(b.get('label', '')).strip()[:80]
                coords = {k: bounded(b[k], 0, 1, float) for k in ('x1','y1','x2','y2')}
                if not label or coords['x2'] <= coords['x1'] or coords['y2'] <= coords['y1']:
                    raise ValueError('Every box needs a label and a positive area')
                cleaned.append({'label': label, **coords})
            photo.update(boxes=cleaned, split=split, reviewed=bool(data.get('reviewed', photo['reviewed'])))
            service.save(); return jsonify(photo)

    @bp.post('/auto-collect')
    def auto_collect():
        data = request.get_json(); cid = data.get('camera')
        if cid not in {c['id'] for c in context['settings']['cameras']}:
            raise ValueError('Choose a configured camera')
        if data.get('split', 'train') not in {'train','val','test'}:
            raise ValueError('Invalid split')
        with service.lock:
            service.state['auto_collect'][cid] = {'enabled': bool(data.get('enabled')), 'interval': bounded(data.get('interval', 30), 10, 3600),
                                                 'confidence': bounded(data.get('confidence', .7), .05, .95, float), 'split': data.get('split','train')}
            service.save()
        return jsonify(ok=True)

    @bp.post('/auto-split')
    def auto_split():
        return jsonify(service.auto_split())

    @bp.post('/datasets')
    def dataset():
        if request.files:
            file = request.files.get('file')
            if not file:
                raise ValueError('Select a YOLO dataset ZIP')
            did = uuid.uuid4().hex[:12]; archive = service.path('downloads/' + did + '.zip')
            archive.parent.mkdir(exist_ok=True); file.save(archive)
            folder = service.path('datasets/' + did)
            try:
                safe_extract(archive, folder)
                result = service.register_dataset(did, folder, file.filename[:100])
            finally:
                archive.unlink(missing_ok=True)
            return jsonify(result)
        return jsonify(service.build_dataset((request.get_json(silent=True) or {}).get('photos')))

    @bp.delete('/datasets/<did>')
    def delete_dataset(did):
        return jsonify(service.delete_dataset(did))

    @bp.get('/datasets/<did>/download')
    def download_dataset(did):
        import yaml
        data = service.item('datasets', did)
        config = yaml.safe_load(service.path(data['yaml']).read_text(encoding='utf-8'))
        archive = service.path('downloads/' + did + '-export.zip'); archive.parent.mkdir(exist_ok=True)
        with service.lock, zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            portable = {'names': config['names']}
            for split in ('train','val','test'):
                if split not in config:
                    continue
                portable[split] = f'{split}/images'
                images = Path(config[split])
                for image in images.rglob('*'):
                    if image.suffix.lower() not in {'.jpg','.jpeg','.png'}:
                        continue
                    relative = image.relative_to(images)
                    parts = list(image.parts); parts[len(images.parts)-1 if images.name == 'images' else len(images.parts)-2] = 'labels'
                    label = Path(*parts).with_suffix('.txt')
                    z.write(image, f'{split}/images/{relative.as_posix()}')
                    z.write(label, f'{split}/labels/{relative.with_suffix(".txt").as_posix()}')
            z.writestr('data.yaml', yaml.safe_dump(portable, allow_unicode=True))
        return send_file(archive, as_attachment=True, download_name=did + '-yolo.zip')

    @bp.post('/train')
    def train():
        data = request.get_json(); service.item('datasets', data.get('dataset'))
        data.update(epochs=bounded(data.get('epochs', 30), 1, 300), imgsz=bounded(data.get('imgsz', 640), 160, 1280), batch=bounded(data.get('batch', 4), 1, 32))
        if data.get('device', 'cpu') not in {'cpu','0'}:
            raise ValueError('Choose CPU or GPU 0')
        data['device'] = data.get('device','cpu')
        data['name'] = str(data.get('name',''))[:100]
        return jsonify(service.start('Local training', lambda jid: service.train(jid, data))), 202

    @bp.post('/models')
    def import_model():
        file = request.files.get('file')
        if not file or not file.filename.lower().endswith('.pt'):
            raise ValueError('Select a compatible Ultralytics detection .pt model')
        target = service.path('downloads/' + uuid.uuid4().hex + '.pt'); target.parent.mkdir(exist_ok=True); file.save(target)
        name = file.filename[:100]
        def work(jid):
            try:
                model = service.register_model(jid, target, name)
                return {'model_id': model['id']}
            finally:
                target.unlink(missing_ok=True)
        try:
            return jsonify(service.start('Import model', work)), 202
        except Exception:
            target.unlink(missing_ok=True); raise

    @bp.post('/models/<mid>/evaluate')
    def evaluate(mid):
        did = request.get_json().get('dataset'); service.item('models', mid); service.item('datasets', did)
        return jsonify(service.start('Model evaluation', lambda jid: service.evaluate(jid, mid, did))), 202

    @bp.post('/models/current')
    def import_current():
        settings = context['settings']
        source = Path(settings['custom_model'])
        if not source.is_absolute():
            source = Path(context['BASE']) / source
        if not source.is_file():
            raise ValueError('The active custom model file is not available on this server')
        classes = settings['custom_classes'][:]
        def work(jid):
            model = service.register_model(jid, source, 'Baseline · ' + source.name, classes=classes)
            return {'model_id': model['id']}
        return jsonify(service.start('Import current model for comparison', work)), 202

    @bp.get('/models/<mid>/download')
    def export_model(mid):
        return send_file(service.path(service.item('models',mid)['file']), as_attachment=True, download_name=mid+'.pt')

    @bp.post('/models/<mid>/activate')
    def activate(mid):
        return jsonify(service.activate(mid, request.get_json().get('cameras', [])))

    @bp.delete('/models/<mid>')
    def delete_model(mid):
        return jsonify(service.delete_model(mid))

    @bp.post('/rollback')
    def rollback():
        return jsonify(service.rollback())

    @bp.post('/jobs/clear')
    def clear_jobs():
        return jsonify(service.clear_jobs())

    @bp.post('/jobs/<jid>/cancel')
    def cancel(jid):
        with service.lock:
            job = service.item('jobs', jid)
            if job['status'] not in {'running', 'queued', 'cancelling'}:
                raise ValueError('Job has already ended')
            service.cancelled.add(jid)
            service.update_job(jid, status='cancelling', message='Cancelling after the current operation')
        return jsonify(ok=True)

    @bp.post('/roboflow/connect')
    def connect():
        data = request.get_json()
        with service.lock:
            if any(j['status'] in {'running','queued','cancelling'} for j in service.state['jobs'].values()):
                raise ValueError('Wait for the current job before changing the connection')
            connection = {'workspace': slug(data.get('workspace','')), 'project': slug(data.get('project','')),
                          'api_key': str(data.get('api_key') or service.connection['api_key']).strip()}
            if not connection['api_key']:
                raise ValueError('Enter a private Roboflow API key')
            service.connection = connection
            atomic_json(service.connection_file, connection); os.chmod(service.connection_file, 0o600)
        result = service.rf()
        return jsonify(ok=True, project=result.get('project', {}).get('name', connection['project']))

    @bp.post('/roboflow/upload')
    def rf_upload():
        ids = list(dict.fromkeys(request.get_json().get('photos', [])))
        if not ids or len(ids) > 500:
            raise ValueError('Select 1–500 photos to upload')
        for pid in ids:
            service.item('photos', pid)
        return jsonify(service.start('Roboflow upload', lambda jid: service.rf_upload(jid, ids))), 202

    @bp.post('/roboflow/dataset')
    def rf_dataset():
        version = bounded(request.get_json().get('version'), 1, 100000)
        return jsonify(service.start('Roboflow dataset download', lambda jid: service.rf_dataset(jid, version))), 202

    @bp.post('/roboflow/model')
    def rf_model():
        model = slug(request.get_json().get('model',''))
        return jsonify(service.start('Roboflow model download', lambda jid: service.rf_model(jid, model))), 202

    @bp.get('/roboflow/status')
    def rf_status():
        version = bounded(request.args.get('version'), 1, 100000)
        data = service.rf(f'/{version}/v2/trainings')
        return jsonify(trainings=[{k: row.get(k) for k in ('id','status','modelType','modelIds')} for row in data.get('trainings',[])])

    @bp.post('/roboflow/train')
    def rf_train():
        data = request.get_json(); version = bounded(data.get('version'), 1, 100000)
        model_type = slug(data['model_type']) if data.get('model_type') else None
        # A single explicit request, never an automatic retry of a potentially billable operation.
        payload = {'epochs': bounded(data.get('epochs',30),1,300)}
        if model_type:
            payload['modelType'] = model_type
        service.rf(f'/{version}/v2/trainings', 'POST', json=payload)
        return jsonify(ok=True, message='Training submitted to Roboflow. Refresh cloud status to follow progress')

    app.config['MAX_CONTENT_LENGTH'] = 1024**3
    app.register_blueprint(bp)
    return service

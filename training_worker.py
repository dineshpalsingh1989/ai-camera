"""Isolated, cancellable YOLO worker. Invoked only by TrainingService."""
import json
import math
import os
from pathlib import Path
import sys


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def run(folder):
    import torch
    import numpy as np
    from ultralytics import YOLO
    # Keep CPU training from consuming every core needed by live cameras.
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2)//2)))
    request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
    write(folder / 'progress.json', {'progress': 1, 'message': 'Loading model'})
    model = YOLO(request['model'])
    if model.task != 'detect':
        raise ValueError('This app accepts Ultralytics object-detection models only; other architectures need a different runtime')
    action = request['action']
    if action == 'inspect':
        if request.get('classes') and 'world' in Path(request['model']).name.lower():
            model.set_classes(request['classes'])
        model.predict(np.zeros((160,160,3), np.uint8), imgsz=160, device='cpu', verbose=False)
        normalized = folder / 'inspected.pt'
        model.save(normalized)
        return {'classes': list(model.names.values()), 'model': str(normalized)}
    device = request.get('device','cpu')
    if device != 'cpu' and not torch.cuda.is_available():
        raise ValueError('CUDA GPU is unavailable. Select CPU for local training')
    import yaml
    data = yaml.safe_load(Path(request['data']).read_text(encoding='utf-8'))
    if action == 'validate' and list(model.names.values()) != data['names']:
        raise ValueError('Model and validation dataset classes differ')
    model_path = request['model']
    if action == 'train':
        epochs = request['epochs']
        def epoch_done(trainer):
            epoch = trainer.epoch + 1
            write(folder / 'progress.json', {'progress': round(epoch / epochs * 85), 'message': f'Training epoch {epoch} / {epochs}'})
        model.add_callback('on_fit_epoch_end', epoch_done)
        model.train(data=request['data'], epochs=epochs, imgsz=request['imgsz'], batch=request['batch'], device=device,
                    workers=0, project=str(folder), name='run', exist_ok=True, plots=False, verbose=False,
                    amp=False, seed=42, deterministic=True, patience=15, cache=False)
        model_path = str(folder / 'run' / 'weights' / 'best.pt')
        model = YOLO(model_path)
    write(folder / 'progress.json', {'progress': 90, 'message': 'Evaluating held-out validation images'})
    metrics = model.val(data=request['data'], split='val', imgsz=request.get('imgsz',640), batch=request.get('batch',1),
                        device=device, workers=0, plots=False, verbose=False, project=str(folder), name='validation')
    values = {'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
              'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map)}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Validation returned invalid metrics. Check your labels and validation data')
    return {'model': model_path, 'classes': list(model.names.values()), 'metrics': values}


if __name__ == '__main__':
    folder = Path(sys.argv[1]).resolve()
    try:
        write(folder / 'result.json', run(folder))
    except Exception as exc:
        write(folder / 'result.json', {'error': str(exc)[:1500]})
        raise

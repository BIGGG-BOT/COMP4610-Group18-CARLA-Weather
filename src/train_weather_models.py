"""Train two independent 50-epoch YOLO11n pilot models and cross-evaluate.

Requires audited CARLA ground-truth data. Never overwrites the original detector.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def audit_dataset(data):
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    design = json.loads((data / 'design.json').read_text(encoding='utf-8'))
    roads, groups, hashes = defaultdict(set), defaultdict(set), defaultdict(set)
    counts, class_counts, pair_keys = Counter(), Counter(), defaultdict(dict)
    for row in manifest:
        weather, split, stem = row['weather'], row['split'], row['stem']
        roads[split].add(row['road_id'])
        groups[split].add(row['group'])
        hashes[split].add(row['image_sha256'])
        key = f'{weather}/{split}'
        counts[key] += 1
        image = data / weather / 'images' / split / f'{stem}.jpg'
        label = data / weather / 'labels' / split / f'{stem}.txt'
        assert sha(image) == row['image_sha256'] and sha(label) == row['label_sha256']
        pixels = cv2.imread(str(image))
        assert pixels is not None and pixels.shape == (450, 800, 3)
        lines = label.read_text().splitlines()
        assert len(lines) == len(row['labels'])
        for line in lines:
            v = list(map(float, line.split()))
            assert len(v) == 5 and v[0] in (0, 1) and all(np.isfinite(v))
            assert all(0 <= x <= 1 for x in v[1:]) and v[3] > 0 and v[4] > 0
            assert v[1] - v[3]/2 >= -1e-6 and v[1] + v[3]/2 <= 1+1e-6
            assert v[2] - v[4]/2 >= -1e-6 and v[2] + v[4]/2 <= 1+1e-6
            class_counts[f'{key}/{int(v[0])}'] += 1
        assert weather not in pair_keys[stem]
        pair_keys[stem][weather] = row
    for split in ('train', 'val', 'test'):
        for other in ('train', 'val', 'test'):
            if split != other:
                assert not roads[split] & roads[other], 'Road leakage'
                assert not groups[split] & groups[other], 'Scenario leakage'
                assert not hashes[split] & hashes[other], 'Exact image duplication across splits'
        for weather in ('clear', 'rain'):
            assert counts[f'{weather}/{split}'] >= 30
            for cid in (0, 1):
                assert class_counts[f'{weather}/{split}/{cid}'] >= 30, 'Insufficient class coverage'
    for pair in pair_keys.values():
        assert set(pair) == {'clear', 'rain'}
        for key in ['split', 'road_id', 'group', 'camera_pose']:
            assert pair['clear'][key] == pair['rain'][key]
        assert pair['clear']['weather_parameters']['precipitation'] == 0
        assert pair['rain']['weather_parameters']['precipitation'] == 100
    result = {'status': 'passed', 'images': len(manifest), 'pairs': len(pair_keys),
              'images_by_split': dict(counts), 'boxes_by_class': dict(class_counts),
              'roads_by_split': {k: sorted(v) for k, v in roads.items()},
              'manifest_sha256': sha(data / 'manifest.json'), 'design_sha256': sha(data / 'design.json'),
              'checks': ['all image and label hashes', 'image decode', 'label bounds', 'both classes in every split',
                         'road and group split isolation', 'no exact duplicate images across splits',
                         'paired camera poses and weather readback'],
              'note': 'Road holdout within one map, not proof of geographical independence or real-world generalisation.'}
    write_json(data / 'audit.json', result)
    return result


def metric_result(result):
    return {'precision': float(result.box.mp), 'recall': float(result.box.mr),
            'mAP50': float(result.box.map50), 'mAP50_95': float(result.box.map),
            'per_class': [{'class_id': int(cid), 'name': result.names[int(cid)],
                           'precision': float(result.box.p[i]), 'recall': float(result.box.r[i]),
                           'AP50': float(result.box.ap50[i]), 'AP50_95': float(result.box.ap[i])}
                          for i, cid in enumerate(result.box.ap_class_index)],
            'speed_ms_per_image': {k: float(v) for k, v in result.speed.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    data = args.dataset.resolve()
    audit = audit_dataset(data)
    print(json.dumps(audit), flush=True)
    if args.audit_only:
        return
    assert torch.cuda.is_available()
    # CARLA should be shut down before calling this script to free VRAM.
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source = ROOT / 'models/yolo_carla_best.pt'
    source_hash = sha(source)
    base = YOLO(str(source))
    initial_backbone = base.model.state_dict()['model.0.conv.weight'].detach().cpu().clone()
    del base
    common = dict(epochs=50, patience=0, imgsz=640, batch=8, workers=0, device=0,
                  cache=False, amp=True, seed=18, deterministic=True, optimizer='AdamW',
                  lr0=.0005, lrf=.1, warmup_epochs=3, weight_decay=.0005, cos_lr=True,
                  mosaic=.2, close_mosaic=10, scale=.2, translate=.05, fliplr=.5,
                  hsv_h=.015, hsv_s=.2, hsv_v=.1, mixup=0., copy_paste=0.,
                  save=True, save_period=10, plots=True, verbose=False)
    experiment = {'status': 'running', 'started': datetime.now().isoformat(), 'source_model': str(source),
                  'source_sha256': source_hash, 'dataset': str(data), 'dataset_audit': audit,
                  'training_config': common, 'gpu': torch.cuda.get_device_name(0),
                  'design': 'Independent clear-only and rain-only fine-tuning from the same third-party YOLO11n checkpoint.',
                  'class_scope': 'vehicle and pedestrian; 4-class pretrained detector transferred into a 2-class head',
                  'selection': 'best.pt selected on each training weather validation split; both models tested on both weather test splits',
                  'limitations': ['synthetic static scenes', 'one map with held-out road IDs', 'visible-mask boxes',
                                  'no trained driving controller', 'no claim of improved closed-loop driving',
                                  'not a clear-versus-clear-plus-rain augmentation experiment',
                                  'upstream pretraining data overlap with Town02 is unknown; holdout applies to this fine-tuning dataset'],
                  'runs': {}, 'test_results': []}
    write_json(out / 'experiment.json', experiment)
    started = time.time()
    try:
        for weather in ('clear', 'rain'):
            print(f'START {weather} 50 EPOCHS', flush=True)
            torch.cuda.reset_peak_memory_stats()
            model = YOLO(str(source))
            t0 = time.time()
            def record_initialization(trainer):
                h = hashlib.sha256()
                for key, tensor in trainer.model.state_dict().items():
                    h.update(key.encode())
                    h.update(tensor.detach().cpu().numpy().tobytes())
                initial = {'weather': weather, 'initial_parameter_sha256': h.hexdigest(),
                           'trainable_parameters': sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                           'names': trainer.model.names}
                if weather == 'rain':
                    first = json.loads((out / 'clear_initialization.json').read_text())
                    assert first['initial_parameter_sha256'] == initial['initial_parameter_sha256'], 'Initial models differ'
                write_json(out / f'{weather}_initialization.json', initial)
            def report_epoch(trainer):
                state = {'stage': 'training', 'weather': weather, 'epoch_completed': trainer.epoch+1,
                         'target_epochs': 50, 'elapsed_s': time.time()-t0,
                         'gpu_allocated_gib': torch.cuda.memory_allocated()/1024**3,
                         'gpu_reserved_gib': torch.cuda.memory_reserved()/1024**3}
                write_json(out / 'progress.json', state)
            model.add_callback('on_fit_epoch_end', report_epoch)
            model.add_callback('on_train_start', record_initialization)
            model.train(data=str(data / f'{weather}.yaml'), project=str(out), name=f'{weather}_50epochs', **common)
            run = Path(model.trainer.save_dir)
            with (run / 'results.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 50, f'Expected 50 completed epochs, got {len(rows)}'
            assert int(float(rows[-1]['epoch'])) == 50
            for row in rows:
                assert all(np.isfinite(float(v)) for v in row.values()), 'Non-finite training metrics'
            best, last = run / 'weights/best.pt', run / 'weights/last.pt'
            assert best.exists() and last.exists()
            trained = YOLO(str(last))
            change = float((trained.model.state_dict()['model.0.conv.weight'].detach().cpu() - initial_backbone).abs().max())
            assert change > 0, 'Backbone weights did not change'
            experiment['runs'][weather] = {'epochs_completed': 50, 'elapsed_s': time.time()-t0,
                                           'run_directory': str(run), 'best': str(best), 'last': str(last),
                                           'best_sha256': sha(best), 'last_sha256': sha(last),
                                           'backbone_max_absolute_change': change,
                                           'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
                                           'last_epoch_metrics': rows[-1]}
            write_json(out / 'experiment.json', experiment)
            del trained, model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            assert sha(source) == source_hash
        checkpoints = {'original_pretrained': source}
        checkpoints.update({f'{w}_finetuned': Path(experiment['runs'][w]['best']) for w in ('clear', 'rain')})
        for model_name, checkpoint in checkpoints.items():
            for weather in ('clear', 'rain'):
                write_json(out / 'progress.json', {'stage': 'test_evaluation', 'model': model_name, 'weather': weather})
                print(f'TEST {model_name} ON {weather}', flush=True)
                model = YOLO(str(checkpoint))
                result = model.val(data=str(data / f'{weather}.yaml'), split='test', imgsz=640,
                                   batch=8, workers=0, device=0, classes=[0, 1],
                                   conf=.001, iou=.7, max_det=300, plots=True,
                                   project=str(out / 'evaluation'), name=f'{model_name}_on_{weather}', verbose=False)
                row = {'model': model_name, 'test_weather': weather, **metric_result(result)}
                experiment['test_results'].append(row)
                write_json(out / 'experiment.json', experiment)
                del model
                torch.cuda.empty_cache()
        assert sha(source) == source_hash
        experiment['status'] = 'complete'
        experiment['elapsed_s'] = time.time()-started
        experiment['finished'] = datetime.now().isoformat()
        write_json(out / 'experiment.json', experiment)
        write_json(out / 'progress.json', {'stage': 'complete', 'elapsed_s': time.time()-started})
        print('ALL TRAINING AND EVALUATION COMPLETE', flush=True)
    except Exception as exc:
        experiment['status'] = 'error'
        experiment['error'] = repr(exc)
        write_json(out / 'experiment.json', experiment)
        raise


if __name__ == '__main__':
    main()

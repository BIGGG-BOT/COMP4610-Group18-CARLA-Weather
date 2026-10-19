"""Independent CARLA instance-mask labels for paired weather fine-tuning.

No detector is used to generate labels. Whole road IDs are held out from training.
Two-class pilot: vehicle and pedestrian; visible-instance bounding boxes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import queue
import random
import time

import carla
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
W, H = 800, 450
SEED = 20261016
NAMES = {0: 'vehicle', 1: 'pedestrian'}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frame(q, target):
    while True:
        item = q.get(timeout=40)
        if item.frame == target:
            return item
        if item.frame > target:
            raise RuntimeError('Camera frame passed requested simulation frame')


def labels_from_mask(bgra):
    # Use installed CARLA enums: 0.9.15 documentation contains older numeric IDs.
    tags = {int(carla.CityObjectLabel.Pedestrians): 1}
    for name in ['Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle']:
        tags[int(getattr(carla.CityObjectLabel, name))] = 0
    semantic = bgra[:, :, 2]
    instance = bgra[:, :, 0].astype(np.uint16) + 256 * bgra[:, :, 1].astype(np.uint16)
    labels = []
    for tag, cid in tags.items():
        ys, xs = np.where(semantic == tag)
        if not len(xs):
            continue
        for iid in np.unique(instance[ys, xs]):
            keep = instance[ys, xs] == iid
            xx, yy = xs[keep], ys[keep]
            x1, x2 = int(xx.min()), int(xx.max()) + 1
            y1, y2 = int(yy.min()), int(yy.max()) + 1
            # Retain tiny objects too; only single-pixel/degenerate raster fragments excluded.
            if len(xx) < 4 or x2 - x1 < 2 or y2 - y1 < 2:
                continue
            labels.append({'class_id': cid, 'box': [x1, y1, x2, y2],
                           'visible_pixels': int(len(xx)), 'instance_id': int(iid), 'semantic_tag': tag})
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-groups', type=int, default=60)
    parser.add_argument('--frames-per-group', type=int, default=10)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'collector_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for weather in ['clear', 'rain']:
        for split in ['train', 'val', 'test']:
            for kind in ['images', 'labels', 'masks']:
                (out / weather / kind / split).mkdir(parents=True)
    (out / 'preview').mkdir()
    rng = random.Random(SEED)
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(120)
    world = client.get_world()
    if world.get_map().name.rsplit('/', 1)[-1] != 'Town02':
        world = client.load_world('Town02')
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = .05
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    road_map = world.get_map()
    library = world.get_blueprint_library()
    camera_actors, scene_actors = [], []
    records = []
    started = time.time()
    try:
        queues = []
        for sensor in ['sensor.camera.rgb', 'sensor.camera.instance_segmentation']:
            bp = library.find(sensor)
            for key, value in [('image_size_x', str(W)), ('image_size_y', str(H)), ('fov', '90')]:
                bp.set_attribute(key, value)
            camera = world.spawn_actor(bp, carla.Transform(carla.Location(z=3)))
            q = queue.Queue()
            camera.listen(q.put)
            camera_actors.append(camera)
            queues.append(q)

        def tick(n=1):
            for _ in range(n):
                fid = world.tick()
                frames = [read_frame(q, fid) for q in queues]
            return frames

        by_road = defaultdict(list)
        for idx, transform in enumerate(road_map.get_spawn_points()):
            wp = road_map.get_waypoint(transform.location)
            if not wp.is_junction:
                by_road[wp.road_id].append((idx, transform))
        roads = sorted(by_road)
        rng.shuffle(roads)
        if len(roads) < 10:
            raise RuntimeError('Insufficient independent road groups')
        ntest = max(2, round(len(roads) * .15))
        nval = max(2, round(len(roads) * .15))
        split_roads = {'test': roads[:ntest], 'val': roads[ntest:ntest+nval], 'train': roads[ntest+nval:]}
        road_to_split = {r: split for split, rr in split_roads.items() for r in rr}
        # Round robin over roads prevents a few long roads monopolising the dataset.
        for values in by_road.values():
            rng.shuffle(values)
        selected = []
        for depth in range(max(map(len, by_road.values()))):
            for road in roads:
                if depth < len(by_road[road]) and len(selected) < args.max_groups:
                    idx, transform = by_road[road][depth]
                    selected.append((road, idx, transform))
        design = {'seed': SEED, 'map': road_map.name, 'names': NAMES, 'split_roads': split_roads,
                  'groups': [{'road_id': r, 'spawn_index': i, 'split': road_to_split[r]} for r, i, _ in selected],
                  'frames_per_group': args.frames_per_group, 'camera': {'width': W, 'height': H, 'fov': 90, 'height_m': 2.4},
                  'labels': 'Visible instance-mask bounding boxes, simulator ground truth, not YOLO predictions',
                  'weather': {'clear': 'ClearNoon', 'rain': 'HardRainNoon'},
                  'limitation': 'Small static-scene synthetic pilot, road-group holdout within Town02; not real-world validation',
                  'source_checkpoint_sha256': sha(ROOT / 'models/yolo_carla_best.pt'),
                  'collector_sha256': sha(Path(__file__)), 'carla_version': client.get_server_version()}
        write_json(out / 'design.json', design)
        print(json.dumps({'groups': len(selected), 'split_roads': split_roads}), flush=True)
        vehicle_bps = sorted([b for b in library.filter('vehicle.*') if b.has_attribute('number_of_wheels') and b.get_attribute('number_of_wheels').as_int() == 4], key=lambda b: b.id)
        walker_bps = sorted(library.filter('walker.pedestrian.*'), key=lambda b: b.id)
        for group_index, (road, spawn_index, transform) in enumerate(selected):
            for actor in reversed(scene_actors):
                actor.destroy()
            scene_actors = []
            split = road_to_split[road]
            wp = road_map.get_waypoint(transform.location)
            forward, right = wp.transform.get_forward_vector(), wp.transform.get_right_vector()
            origin = wp.transform.location
            specs = []
            # Each group changes identities and arrangements; static actors enable exact paired geometry.
            for k in range(5):
                bp = rng.choice(walker_bps)
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'true')
                distance, lateral = rng.uniform(9, 40), rng.uniform(-6, 6)
                location = origin + forward * distance + right * lateral
                # Town02 road/sidewalk mesh can sit above OpenDRIVE z=0.
                # Spawn above the surface and allow gravity to settle the walker.
                location.z += 1.0
                actor = world.try_spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=rng.uniform(-180, 180))))
                if actor:
                    actor.apply_control(carla.WalkerControl(speed=0.0))
                    scene_actors.append(actor)
                    specs.append({'id': actor.id, 'blueprint': bp.id, 'kind': 'pedestrian', 'distance': distance, 'lateral': lateral})
            for k in range(3):
                bp = rng.choice(vehicle_bps)
                if bp.has_attribute('color'):
                    colors = bp.get_attribute('color').recommended_values
                    if colors:
                        bp.set_attribute('color', rng.choice(colors))
                distance = 18 + k * 13 + rng.uniform(-3, 3)
                lateral = rng.choice([-3.5, 0, 3.5])
                location = origin + forward * distance + right * lateral
                location.z += .35
                actor = world.try_spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=transform.rotation.yaw)))
                if actor:
                    actor.apply_control(carla.VehicleControl(brake=1.0))
                    scene_actors.append(actor)
                    specs.append({'id': actor.id, 'blueprint': bp.id, 'kind': 'vehicle', 'distance': distance, 'lateral': lateral})
            # Walkers must retain their character movement/physics to be rendered.
            # Settle vehicles onto the ground before freezing their rigid bodies.
            tick(40)
            for actor in scene_actors:
                if actor.type_id.startswith('vehicle.'):
                    actor.set_simulate_physics(False)
            write_json(out / f'group_{group_index:03d}.json', {'road': road, 'spawn_index': spawn_index, 'actors': specs})
            poses = []
            for j in range(args.frames_per_group):
                loc = origin + forward * (j * .65 - 2.0) + right * rng.uniform(-.35, .35)
                loc.z += 2.4
                poses.append(carla.Transform(loc, carla.Rotation(pitch=-2, yaw=transform.rotation.yaw + rng.uniform(-5, 5))))
            weather_order = ['clear', 'rain'] if group_index % 2 == 0 else ['rain', 'clear']
            for weather in weather_order:
                preset = getattr(carla.WeatherParameters, design['weather'][weather])
                world.set_weather(preset)
                for camera in camera_actors:
                    camera.set_transform(poses[0])
                tick(25)
                actual = world.get_weather()
                params = {n: getattr(actual, n) for n in dir(actual) if not n.startswith('_') and isinstance(getattr(actual, n), (float, int))}
                assert all(abs(v - getattr(preset, k)) < 1e-4 for k, v in params.items()), 'Weather readback mismatch'
                for j, pose in enumerate(poses):
                    for camera in camera_actors:
                        camera.set_transform(pose)
                    rgb, seg = tick(4)
                    assert rgb.frame == seg.frame
                    image = np.frombuffer(rgb.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3].copy()
                    mask = np.frombuffer(seg.raw_data, np.uint8).reshape(H, W, 4).copy()
                    labels = labels_from_mask(mask)
                    stem = f'g{group_index:03d}_r{road}_s{spawn_index}_f{j:02d}'
                    image_path = out / weather / 'images' / split / f'{stem}.jpg'
                    label_path = out / weather / 'labels' / split / f'{stem}.txt'
                    assert cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    assert cv2.imwrite(str(out / weather / 'masks' / split / f'{stem}.png'), mask)
                    lines = []
                    for item in labels:
                        x1, y1, x2, y2 = item['box']
                        lines.append(f"{item['class_id']} {(x1+x2)/(2*W):.8f} {(y1+y2)/(2*H):.8f} {(x2-x1)/W:.8f} {(y2-y1)/H:.8f}")
                    label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
                    if j == 0:
                        preview = image.copy()
                        for item in labels:
                            x1, y1, x2, y2 = item['box']
                            color = (0, 220, 0) if item['class_id'] == 0 else (0, 180, 255)
                            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 1)
                            cv2.putText(preview, NAMES[item['class_id']], (x1, max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
                        cv2.imwrite(str(out / 'preview' / f'{stem}_{weather}.jpg'), preview)
                    records.append({'stem': stem, 'weather': weather, 'split': split, 'road_id': road, 'group': group_index,
                                    'frame': rgb.frame, 'labels': labels, 'image_sha256': sha(image_path), 'label_sha256': sha(label_path),
                                    'weather_parameters': params, 'camera_pose': {'x': pose.location.x, 'y': pose.location.y, 'z': pose.location.z,
                                    'yaw': pose.rotation.yaw, 'pitch': pose.rotation.pitch}})
            write_json(out / 'progress.json', {'finished_groups': group_index + 1, 'total_groups': len(selected), 'images': len(records), 'seconds': time.time()-started})
            print(f'GROUP {group_index+1}/{len(selected)} complete; {len(records)} images; {time.time()-started:.1f}s', flush=True)
        write_json(out / 'manifest.json', records)
        for weather in ['clear', 'rain']:
            yaml = f"path: {out.as_posix()}/{weather}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: vehicle\n  1: pedestrian\n"
            (out / f'{weather}.yaml').write_text(yaml, encoding='utf-8')
        write_json(out / 'summary.json', {'status': 'collected', 'images': len(records), 'counts': dict(Counter(f"{r['weather']}/{r['split']}" for r in records)), 'elapsed_s': time.time()-started})
        print('COLLECTION COMPLETE', flush=True)
    finally:
        for actor in reversed(camera_actors + scene_actors):
            try:
                if hasattr(actor, 'stop'):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original)


if __name__ == '__main__':
    main()

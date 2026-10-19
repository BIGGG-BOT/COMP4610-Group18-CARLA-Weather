"""Group 18: RGB/YOLO longitudinal control with map-assisted steering.

Pedestrian ground truth is used ONLY by scenario generation and evaluation.
The safety controller receives detector boxes, never pedestrian actor positions.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
import carla
import cv2
import numpy as np
from ultralytics import YOLO

DT = 0.05
WIDTH, HEIGHT, FOV = 800, 450, 90
FOCAL = WIDTH / (2 * math.tan(math.radians(FOV / 2)))


def safety_decision(detections, stop_distance=20.0, corridor_half_width=1.9):
    """Single-frame baseline. Geometry is approximate monocular box geometry."""
    hazards = []
    for det in detections:
        name = det['class_name']
        if name not in ('pedestrian', 'vehicle'):
            continue
        x1, y1, x2, y2 = det['box']
        physical_height = 1.7 if name == 'pedestrian' else 1.5
        distance = physical_height * FOCAL / max(y2 - y1, 1)
        lateral = ((x1 + x2) / 2 - WIDTH / 2) * distance / FOCAL
        det['estimated_distance_m'] = distance
        det['estimated_lateral_m'] = lateral
        if y2 > HEIGHT / 2 and distance < stop_distance and abs(lateral) < corridor_half_width:
            hazards.append(det)
    return bool(hazards), hazards


def choose_route(road_map, requested_length=70.0):
    """Deterministically select an intersection-free straight stretch."""
    for index, spawn in enumerate(road_map.get_spawn_points()):
        first = road_map.get_waypoint(spawn.location)
        if first.is_junction:
            continue
        points = [first]
        for _ in range(int(requested_length / 2)):
            choices = points[-1].next(2.0)
            if len(choices) != 1:
                break
            nxt = choices[0]
            yaw_error = (nxt.transform.rotation.yaw - first.transform.rotation.yaw + 180) % 360 - 180
            if nxt.is_junction or abs(yaw_error) > 3 or nxt.lane_id != first.lane_id:
                break
            points.append(nxt)
        if len(points) == int(requested_length / 2) + 1:
            return index, spawn, points
    raise RuntimeError('No suitable straight route found in Town01.')


def transform_dict(transform):
    return {'x': transform.location.x, 'y': transform.location.y, 'z': transform.location.z,
            'yaw': transform.rotation.yaw}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['empty', 'pedestrian'], default='pedestrian')
    parser.add_argument('--timeout', type=float, default=60.0, help='Simulation seconds')
    parser.add_argument('--speed', type=float, default=15.0, help='Target km/h')
    parser.add_argument('--seed', type=int, default=18)
    parser.add_argument('--weather', choices=['ClearNoon', 'HardRainNoon'], default='ClearNoon')
    parser.add_argument('--scenario-id', choices=['S01','S02','S03','S04','S05'], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    scenario_file = ROOT / 'weather_suite_scenarios.json'
    suite_design = json.loads(scenario_file.read_text(encoding='utf-8'))
    scenario_spec = next(s for s in suite_design['scenarios'] if s['id'] == args.scenario_id)
    direction = scenario_spec['direction']
    np.random.seed(args.seed)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.scenario_id + '_' + args.weather
    out = args.output_root / run_id
    out.mkdir(parents=True)
    (out / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out / 'keyframes').mkdir()
    model_path = ROOT / 'models' / 'yolo_carla_best.pt'
    model = YOLO(str(model_path))
    model.predict(np.zeros((HEIGHT, WIDTH, 3), np.uint8), device=0, imgsz=640, verbose=False)
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(120)
    world = client.get_world()
    original_settings = world.get_settings()
    actors, rows, box_rows, collisions, events = [], [], [], [], []
    writer = None
    status, error = 'incomplete', None
    reached = stopped = resumed = crossing_started = crossing_finished = False
    first_frame = 0
    wall_start = None
    try:
        if world.get_map().name.rsplit('/', 1)[-1] != 'Town01':
            world = client.load_world('Town01')
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        world = client.reload_world(False)
        world.set_weather(getattr(carla.WeatherParameters, args.weather))
        applied_weather = world.get_weather()
        weather_parameters = {name: getattr(applied_weather, name) for name in dir(applied_weather)
                              if not name.startswith('_') and isinstance(getattr(applied_weather, name), (float, int))}
        road_map = world.get_map()
        route_index, spawn, waypoints = choose_route(road_map)
        origin = waypoints[0].transform.location
        forward = waypoints[0].transform.get_forward_vector()
        right = waypoints[0].transform.get_right_vector()
        route_length = 70.0
        crossing_distance = 34.0
        crossing_center = origin + forward * crossing_distance
        library = world.get_blueprint_library()
        bp = library.find('vehicle.tesla.model3')
        bp.set_attribute('role_name', 'group18_visual_baseline')
        if bp.has_attribute('color'):
            bp.set_attribute('color', '30,80,180')
        vehicle = world.spawn_actor(bp, spawn)
        actors.append(vehicle)
        vehicle.set_autopilot(False)
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        pedestrian = None
        if args.scenario == 'pedestrian':
            walker_bp = library.find('walker.pedestrian.0001')
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            walker_location = crossing_center - right * (3.2 * direction)
            # Walker transform is at its feet, unlike the vehicle spawn offset.
            walker_location.z = origin.z + 0.1
            pedestrian = world.spawn_actor(walker_bp, carla.Transform(walker_location))
            actors.append(pedestrian)
            pedestrian.apply_control(carla.WalkerControl(speed=0.0))
        collision_sensor = world.spawn_actor(library.find('sensor.other.collision'),
                                            carla.Transform(), attach_to=vehicle)
        actors.append(collision_sensor)
        def on_collision(event):
            if first_frame and event.frame >= first_frame:
                impulse = event.normal_impulse
                collisions.append({'frame': event.frame, 'other': event.other_actor.type_id,
                                   'impulse': math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)})
        collision_sensor.listen(on_collision)
        camera_bp = library.find('sensor.camera.rgb')
        for key, value in [('image_size_x', str(WIDTH)), ('image_size_y', str(HEIGHT)),
                           ('fov', str(FOV)), ('sensor_tick', '0.0')]:
            camera_bp.set_attribute(key, value)
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)),
                                   attach_to=vehicle)
        actors.append(camera)
        frames = queue.Queue()
        camera.listen(frames.put)
        config = {'run_id': run_id, 'scenario': args.scenario, 'weather': args.weather, 'weather_parameters': weather_parameters,
                  'seed': args.seed, 'map': road_map.name, 'spawn_index': route_index,
                  'scenario_id': args.scenario_id, 'scenario_spec': scenario_spec,
                  'scenario_design_sha256': hashlib.sha256(scenario_file.read_bytes()).hexdigest(),
                  'start': transform_dict(spawn), 'route_length_m': route_length,
                  'route': [transform_dict(w.transform) for w in waypoints],
                  'crossing_at_m': crossing_distance, 'crossing_start_lateral_m': -3.2 * direction,
                  'crossing_end_lateral_m': 3.2 * direction, 'walker_spawn_z_m': origin.z + 0.1, 'walker_speed_m_s': scenario_spec['walker_speed_m_s'],
                  'crossing_trigger_remaining_m': scenario_spec['trigger_remaining_m'], 'target_speed_kmh': args.speed,
                  'timeout_sim_s': args.timeout, 'fixed_delta_seconds': DT,
                  'camera': {'width': WIDTH, 'height': HEIGHT, 'fov': FOV, 'mount_x': 1.5, 'mount_z': 2.4},
                  'detector': {'weights': model_path.name, 'sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
                               'confidence': 0.35, 'imgsz': 640, 'device': 'cuda:0'},
                  'controller': {'stop_distance_estimate_m': 20, 'corridor_half_width_estimate_m': 1.9,
                                 'safety_input': 'current RGB detections only; no actor ground truth',
                                 'navigation_input': 'map waypoints and ego pose; not purely visual driving',
                                 'temporal_detection_hold': False},
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'carla_version': client.get_server_version()}
        writer = cv2.VideoWriter(str(out / 'drive.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                 1 / DT, (WIDTH, HEIGHT + 96))
        if not writer.isOpened():
            raise RuntimeError('Could not create the video file.')
        # Settle actors and camera auto exposure with the ego brake held.
        for _ in range(30):
            frame_id = world.tick()
            while frames.get(timeout=30).frame != frame_id:
                pass
        # Read back only after synchronous settling: immediate set/get can record
        # the map's previous weather before the requested change becomes active.
        requested_weather = getattr(carla.WeatherParameters, args.weather)
        applied_weather = world.get_weather()
        weather_parameters = {name: getattr(applied_weather, name) for name in dir(applied_weather)
                              if not name.startswith('_') and isinstance(getattr(applied_weather, name), (float, int))}
        expected_weather = {name: getattr(requested_weather, name) for name in weather_parameters}
        if not all(math.isclose(value, expected_weather[name], abs_tol=1e-4, rel_tol=1e-5)
                   for name, value in weather_parameters.items()):
            raise RuntimeError('Active weather does not match requested preset after settling.')
        config['weather_parameters'] = weather_parameters
        config['weather_verified_after_settling'] = True
        save_json(out / 'config.json', config)
        wall_start = time.perf_counter()
        last_state = None
        saved_keys = set()
        min_center_distance = float('inf')
        max_cross_track = 0.0
        for step in range(int(args.timeout / DT)):
            frame_id = world.tick()
            if not first_frame:
                first_frame = frame_id
            while True:
                frame = frames.get(timeout=30)
                if frame.frame == frame_id:
                    break
            sim_t = step * DT
            bgr = np.frombuffer(frame.raw_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy()
            if step in (0, 80, 100, 120, 160, 200):
                cv2.imwrite(str(out / 'keyframes' / f'sim_{sim_t:05.2f}_raw.png'), bgr)
            pose = vehicle.get_transform()
            displacement = pose.location - origin
            progress = displacement.x * forward.x + displacement.y * forward.y
            cross_track = displacement.x * right.x + displacement.y * right.y
            max_cross_track = max(max_cross_track, abs(cross_track))
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            # These actor coordinates belong to the scenario/evaluation, never safety_decision.
            gt_distance, gt_lateral = None, None
            if pedestrian is not None:
                ped_location = pedestrian.get_location()
                delta = ped_location - crossing_center
                gt_lateral = delta.x * right.x + delta.y * right.y
                gt_distance = pose.location.distance(ped_location)
                min_center_distance = min(min_center_distance, gt_distance)
                if not crossing_started and progress >= crossing_distance - scenario_spec['trigger_remaining_m']:
                    crossing_started = True
                    events.append({'event': 'crossing_started', 'sim_s': sim_t, 'frame': frame_id})
                if crossing_started and not crossing_finished:
                    if gt_lateral * direction >= 3.2:
                        crossing_finished = True
                        pedestrian.apply_control(carla.WalkerControl(speed=0.0))
                        events.append({'event': 'crossing_finished', 'sim_s': sim_t, 'frame': frame_id})
                    else:
                        pedestrian.apply_control(carla.WalkerControl(direction=right * direction, speed=scenario_spec['walker_speed_m_s']))
            inference_start = time.perf_counter()
            result = model.predict(bgr, device=0, imgsz=640, conf=0.35, verbose=False)[0]
            inference_ms = (time.perf_counter() - inference_start) * 1000
            detections = []
            for xyxy, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                               result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                detections.append({'class_name': result.names[int(cls)], 'confidence': confidence, 'box': xyxy})
            hazard, hazard_boxes = safety_decision(detections)
            for det in detections:
                box_rows.append({'frame': frame_id, 'sim_s': sim_t, 'class_name': det['class_name'],
                                 'confidence': det['confidence'], 'x1': det['box'][0], 'y1': det['box'][1],
                                 'x2': det['box'][2], 'y2': det['box'][3],
                                 'estimated_distance_m': det.get('estimated_distance_m'),
                                 'estimated_lateral_m': det.get('estimated_lateral_m'),
                                 'hazard': det in hazard_boxes})
            # Map-assisted look-ahead steering; autopilot is explicitly disabled.
            lookahead = min(route_length, max(0, progress) + 6.0)
            target = origin + forward * lookahead
            desired_yaw = math.atan2(target.y - pose.location.y, target.x - pose.location.x)
            error_yaw = (desired_yaw - math.radians(pose.rotation.yaw) + math.pi) % (2 * math.pi) - math.pi
            steer = float(np.clip(1.4 * error_yaw, -0.6, 0.6))
            throttle, brake = 0.0, 0.0
            if progress >= route_length - 2.0:
                reached = True
                state, brake = 'GOAL', 1.0
            elif hazard:
                state, brake = ('WAIT' if speed < 0.2 else 'BRAKE'), 1.0
            else:
                state = 'CRUISE'
                speed_error = args.speed / 3.6 - speed
                throttle = float(np.clip(0.15 + 0.35 * speed_error, 0.0, 0.6))
                if speed_error < -0.3:
                    throttle, brake = 0.0, min(0.3, -0.15 * speed_error)
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
            if state != last_state:
                events.append({'event': 'state', 'state': state, 'sim_s': sim_t, 'frame': frame_id,
                               'progress_m': progress, 'speed_kmh': speed * 3.6})
                print(f'{sim_t:5.2f}s {state:6} progress={progress:5.1f}m speed={speed*3.6:4.1f}km/h', flush=True)
                last_state = state
            if state == 'WAIT':
                stopped = True
            if stopped and not hazard and speed > 1.0:
                resumed = True
            nearest_estimate = min((d['estimated_distance_m'] for d in hazard_boxes), default=None)
            rows.append({'frame': frame_id, 'camera_frame': frame.frame, 'sim_s': sim_t,
                         'progress_m': progress, 'cross_track_m': cross_track, 'speed_kmh': speed * 3.6,
                         'throttle': throttle, 'brake': brake, 'steer': steer, 'state': state,
                         'hazard': hazard, 'hazard_count': len(hazard_boxes), 'detections': len(detections),
                         'hazard_distance_estimate_m': nearest_estimate, 'yolo_wall_ms': inference_ms,
                         'pedestrian_gt_center_distance_m_eval_only': gt_distance,
                         'pedestrian_gt_lateral_m_eval_only': gt_lateral})
            annotated = result.plot()
            canvas = np.zeros((HEIGHT + 96, WIDTH, 3), dtype=np.uint8)
            canvas[:HEIGHT] = annotated
            color = (70, 220, 70) if state in ('CRUISE', 'GOAL') else (40, 180, 255)
            cv2.putText(canvas, f'{args.scenario_id} | {args.weather} | {state} | sim {sim_t:.2f}s',
                        (12, HEIGHT + 26), cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2)
            cv2.putText(canvas, f'{speed*3.6:.1f} km/h | progress {progress:.1f}/70m | brake {brake:.1f}',
                        (12, HEIGHT + 55), cv2.FONT_HERSHEY_SIMPLEX, .60, (240, 240, 240), 1)
            cv2.putText(canvas, 'YOLO -> speed/brake | map-assisted steering | video uses simulation time',
                        (12, HEIGHT + 81), cv2.FONT_HERSHEY_SIMPLEX, .48, (180, 180, 180), 1)
            writer.write(canvas)
            key = 'start' if step == 0 else ('goal' if reached else ('resumed' if resumed else state.lower()))
            if key not in saved_keys:
                cv2.imwrite(str(out / 'keyframes' / f'{len(saved_keys):02d}_{key}.jpg'), canvas)
                cv2.imwrite(str(out / 'keyframes' / f'{len(saved_keys):02d}_{key}_raw.jpg'), bgr)
                saved_keys.add(key)
            if reached:
                status = 'completed'
                break
            if collisions:
                status = 'collision'
                break
            if abs(cross_track) > 2.0:
                status = 'left_route'
                break
        else:
            status = 'timeout'
        # Flush collision callbacks for the last applied control without changing the result log.
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        world.tick()
        if collisions:
            status = 'collision'
    except Exception as exc:
        error, status = repr(exc), 'error'
        raise
    finally:
        wall_elapsed = time.perf_counter() - wall_start if wall_start is not None else None
        if writer is not None:
            writer.release()
        for actor in reversed(actors):
            try:
                if actor.is_alive:
                    if hasattr(actor, 'stop'):
                        actor.stop()
                    actor.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original_settings)
        for filename, values in [('driving_log.csv', rows), ('detections.csv', box_rows)]:
            if values:
                with (out / filename).open('w', newline='', encoding='utf-8-sig') as handle:
                    csv_writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                    csv_writer.writeheader()
                    csv_writer.writerows(values)
        save_json(out / 'events.json', events)
        save_json(out / 'collisions.json', collisions)
        summary = {'run_id': run_id, 'scenario_id': args.scenario_id, 'scenario': args.scenario, 'weather': args.weather, 'status': status,
                   'success_reached_goal_without_collision': reached and not collisions and status == 'completed',
                   'reached_goal': reached, 'collision_events': len(collisions),
                   'pedestrian_crossing_started': crossing_started, 'pedestrian_crossing_finished': crossing_finished,
                   'stopped_for_detection': stopped, 'resumed_after_stop': resumed,
                   'sim_duration_s': rows[-1]['sim_s'] if rows else 0,
                   'frames': len(rows), 'wall_duration_s': wall_elapsed,
                   'wall_processing_fps': len(rows) / wall_elapsed if wall_elapsed else None,
                   'mean_yolo_wall_ms': float(np.mean([r['yolo_wall_ms'] for r in rows])) if rows else None,
                   'max_cross_track_m': max((abs(r['cross_track_m']) for r in rows), default=None),
                   'minimum_pedestrian_center_distance_m_eval_only': min((r['pedestrian_gt_center_distance_m_eval_only']
                         for r in rows if r['pedestrian_gt_center_distance_m_eval_only'] is not None), default=None),
                   'error': error, 'output_directory': str(out)}
        save_json(out / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['success_reached_goal_without_collision'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

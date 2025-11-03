import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

CAPTURE_DIR = Path("/home/namtt/WorkSpace/VisualLocalization/capture")
DEVICES = ['ios', 'hl', 'spot']
SCENES = ['arche_d2']
BENCHMARKING_DIR = 'benchmarking_arche'

def read_poses(file_path):
    poses = defaultdict()
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip() or '#' in line:
                continue
            values = line.strip().split(', ')
            try:
                timestamp = values[0]
                device = values[1]
                quaternion = np.array([float(v) for v in values[2:6]])  # qw, qx, qy, qz
                translation = np.array([float(v) for v in values[6:9]])  # tx, ty, tz
                poses[(timestamp, device)] = (quaternion, translation)
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")
                continue
    return poses

def load_poses(query, map):
    gtr_poses_path = GTR_DIR / f"{query}_query/proc/alignment_trajectories.txt"
    bench_poses_path = BENCH_DIR / f"{query}_query/{map}_map/superpoint/lightglue/megaloc-10/triangulation/single_image/poses.txt"
    if query in ["spot", "hl"]:
        bench_poses_path = BENCH_DIR / f"{query}_query/{map}_map/superpoint/lightglue/megaloc-10/triangulation/rig/poses.txt"
    gtr_pose = read_poses(gtr_poses_path)
    bench_pose = read_poses(bench_poses_path)
    return gtr_pose, bench_pose

def translation_error(T_gt, T_pre):
    T_gt = np.array(T_gt, dtype=float)
    T_pre = np.array(T_pre, dtype=float)
    return float(np.linalg.norm(T_gt - T_pre))

def rotation_error(Q_gt, Q_pre):
    R1 = R.from_quat([Q_gt[1], Q_gt[2], Q_gt[3], Q_gt[0]]).as_matrix()  # Convert to x, y, z, w
    R2 = R.from_quat([Q_pre[1], Q_pre[2], Q_pre[3], Q_pre[0]]).as_matrix()
    R_err = R1.T @ R2
    trace = np.trace(R_err)
    trace = np.clip(trace, -1.0, 3.0)  # Numerical stability
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)

def evaluate_device(capture_dir, location, query_device, map_device, rotation_threshold, translation_threshold,
                   benchmarking_dir, local_feature_method, global_feature_method, matching_method):
    print(f"Evaluating {location} - Query: {query_device}, Map: {map_device}")
    
    location_dir = location.upper()

    gt_path = f"{capture_dir}/{location_dir}/sessions/{query_device}_query/proc/alignment_trajectories.txt"
    gt_poses = read_poses(gt_path)
    
    if query_device == "ios":
        device_type = "single_image"
    else:
        device_type = "rig"
    pre_path = f"{capture_dir}/{location_dir}/{benchmarking_dir}/pose_estimation/{query_device}_query/{map_device}_map/{local_feature_method}/{matching_method}/{global_feature_method}/triangulation/{device_type}/poses.txt"
    pre_poses = read_poses(pre_path)
    
    total_queries = len(gt_poses)
    total_correct = 0

    gt_keys = set(gt_poses.keys())
    pre_keys = set(pre_poses.keys())

    matched_keys = gt_keys.intersection(pre_keys)
    num_matched = len(matched_keys)
    
    for key in matched_keys:
        Q_gt, T_gt = gt_poses[key]
        Q_pre, T_pre = pre_poses[key]
        
        rot_err = rotation_error(Q_gt, Q_pre)
        trans_err = translation_error(T_gt, T_pre)
        if rot_err <= rotation_threshold and trans_err <= translation_threshold:
            total_correct += 1
    
    correct_pose = (total_correct / num_matched * 100) if total_queries > 0 else 0.0
    result = {
        "location": location,
        "query_device": query_device,
        "map_device": map_device,
        "total_queries": total_queries,
        "matched_queries": num_matched,
        "total_correct": total_correct,
        "correct_pose": correct_pose
    }
    return result

def evaluate_all(capture_dir, scenes, devices_query, devices_map, rotation_threshold, translation_threshold,
                 benchmarking_dir, local_feature_method, global_feature_method, matching_method, output_dir="evaluation_results"):
    all_results = []
    for scene in scenes:
        results = []
        result_dir = f"{capture_dir}/{scene.upper()}/{benchmarking_dir}/{output_dir}"
        os.makedirs(result_dir, exist_ok=True)
        for query_device in devices_query:
            for map_device in devices_map:
                print(f"Evaluating Scene: {scene}, Query Device: {query_device}, Map Device: {map_device}")
                result = evaluate_device(capture_dir, scene, query_device, map_device,
                                         rotation_threshold, translation_threshold,
                                         benchmarking_dir, local_feature_method,
                                         global_feature_method, matching_method)
                results.append(result)
                result_file = f"{result_dir}/eval_{query_device}_to_{map_device}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=4)
        df = pd.DataFrame(results)
        pivot_table = df.pivot("query_device", "map_device", "correct_pose")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Correct Pose Percentage Heatmap")
        plt.ylabel("Query Device")
        plt.xlabel("Map Device")
        plt.savefig(f"{result_dir}/evaluation_heatmap.png")
        all_results.extend(results)
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual Localization Performance")
    parser.add_argument('--capture_dir', type=str, default=str(CAPTURE_DIR), help='Path to the capture directory')
    parser.add_argument('--rotation_threshold', type=float, default=5.0, help='Rotation error threshold in degrees')
    parser.add_argument('--translation_threshold', type=float, default=0.5, help='Translation error threshold in meters')
    parser.add_argument('--local_feature_method', type=str, default='superpoint', help='Local feature method used')
    parser.add_argument('--global_feature_method', type=str, default='megaloc-10', help='Global feature method used')
    parser.add_argument('--matching_method', type=str, default='lightglue', help='Feature matching method used')
    parser.add_argument('--benchmarking_dir', type=str, default=BENCHMARKING_DIR, help='Benchmarking directory name')
    parser.add_argument('--scenes', nargs='+', default=SCENES, help='Scenes to evaluate')
    parser.add_argument('--devices_map', nargs='+', default=DEVICES, help='Map devices to evaluate')
    parser.add_argument('--devices_query', nargs='+', default=DEVICES, help='Query devices to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    args = parser.parse_args()

    print("Starting cross-device pose estimation evaluation")
    print(f"Configuration:")
    print(f"  Capture dir: {args.capture_dir}")
    print(f"  Benchmarking dir: {args.benchmarking_dir}")
    print(f"  Local feature method: {args.local_feature_method}")
    print(f"  Matching method: {args.matching_method}")
    print(f"  Global feature method: {args.global_feature_method}")
    print(f"  Scenes: {args.scenes}")
    print(f"  Map devices: {args.devices_map}")
    print(f"  Query devices: {args.devices_query}")
    print(f"  Position threshold: {args.rotation_threshold}")
    print(f"  Rotation threshold: {args.translation_threshold}")

    evaluate_all(args.capture_dir, args.scenes, args.devices_query, args.devices_map,
                 args.rotation_threshold, args.translation_threshold, args.benchmarking_dir,
                 args.local_feature_method, args.global_feature_method, args.matching_method, 
                 args.output_dir)
if __name__ == "__main__":
    main()
from pathlib import Path
import random
import os
import shutil
import collections

def read_images_txt(path):
    lines_dict = collections.defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if "#" in line:
                lines_dict['column_name'] = line.strip()
                continue
            timestamp = line.strip().split(", ")[0]
            sensor_id = line.strip().split(", ")[1]
            path = line.strip().split(", ")[2]
            lines_dict[timestamp].append({
                'sensor_id': sensor_id,
                'path': path,
                'line': line
            })
    return lines_dict

def write_images_txt(images_dict, timestamps, out_path):
    with open(out_path, "w") as f:
        if "column_name" in images_dict:
            f.write(images_dict['column_name']+'\n')
        for ts in timestamps:
            for line in images_dict[ts]:
                f.write(line['line'])
    
def select_random(images_dict, map_ratio=0.1, num_query=10, step=3):
    timestamps = [k for k in images_dict.keys() if k != "column_name"]
    timestamps.sort()
    toltal = len(timestamps)
    num_map = max(1, int(toltal*map_ratio))
    map_timestamps = timestamps[::step][:num_map]
    
    query_check = list(set(timestamps) - set(map_timestamps))
    query_timestamps = random.sample(query_check, num_query)
    
    return map_timestamps, query_timestamps

def copy_txt_file(src_dir, tar_m_dir, tar_q_dir, file_name):
    name = f"{file_name}.txt"
    if os.path.exists(src_dir / name):
        shutil.copy(src_dir / name, tar_m_dir / name)
        shutil.copy(src_dir / name, tar_q_dir / name)
        
def cp_images(images_dict, timestamps, src_dir, dst_dir):
    for ts in timestamps:
        for item in images_dict[ts]:
            rel_img_path = src_dir / item['path']
            dst_img_path = dst_dir / item['path']
            dst_img_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(rel_img_path, dst_img_path)
            
def read_trajectories_txt(path):
    dicts = collections.defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or "#" in line:
                continue
            parts = line.strip().split(", ")
            timestamp = parts[0]
            device_id = parts[1]
            dicts[timestamp] = device_id
    return dicts

def create_keyframe_query(trajectories_path, q_keyframe_path, query_dict):
    trajectories_dict = read_trajectories_txt(trajectories_path)
    with open(q_keyframe_path, "w") as f:
        for ts in query_dict:
            line = f"{ts}, {trajectories_dict[ts]}\n"
            f.write(line)

def copy(src_dir, m_dir, q_dir, map_ratio, num_query, step):
    src_proc_dir = src_dir / "proc"
    m_proc_dir = m_dir / "proc"
    q_proc_dir = q_dir / "proc"
    src_data_dir = src_dir / 'raw_data'
    m_data_dir = m_dir / 'raw_data'
    q_data_dir = q_dir / 'raw_data'
    os.makedirs(m_proc_dir, exist_ok=True)
    os.makedirs(q_proc_dir, exist_ok=True)
    
    images_map_src = read_images_txt(src_dir / "images.txt")
    map_ts, query_ts = select_random(images_map_src, map_ratio, num_query, step)
    write_images_txt(images_map_src, map_ts, m_dir / 'images.txt')
    write_images_txt(images_map_src, query_ts, q_dir / 'images.txt')
    
    copy_txt_file(src_dir, m_dir, q_dir, 'sensors')
    copy_txt_file(src_dir, m_dir, q_dir, 'trajectories')
    copy_txt_file(src_dir, m_dir, q_dir, 'rigs')
    copy_txt_file(src_proc_dir, m_proc_dir, q_proc_dir, 'subsessions')
    
    cp_images(images_map_src, map_ts, src_data_dir, m_data_dir)
    cp_images(images_map_src, query_ts, src_data_dir, q_data_dir)
    
    create_keyframe_query(src_dir/'trajectories.txt', q_proc_dir/'keyframes_pruned_subsampled.txt', query_ts)

def extract_location(src_dir, cp_dir, map_ratio, num_query=10, step=3):
    src_session_dir = src_dir / "sessions"
    cp_session_dir = cp_dir / "sessions"
    os.makedirs(cp_session_dir, exist_ok=True)
    all_sessions = ["ios", "spot", "hl"]
    for session in all_sessions:
        src_m_dir = src_session_dir / f"{session}_map"
        cp_q_dir = cp_session_dir / f"{session}_query"
        cp_m_dir = cp_session_dir / f"{session}_map"
        os.makedirs(cp_q_dir, exist_ok=True)
        os.makedirs(cp_m_dir, exist_ok=True)
        copy(src_m_dir, cp_m_dir, cp_q_dir, map_ratio, num_query, step)
        print(f"===>Finished copy for session: {session}.")
import os
import yaml
import shutil
import argparse
from pathlib import Path
from functions import extract_location

def param_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cross valid data from capture map.")
    parser.add_argument("--config", default="./create_cross_valid/config.yaml", type=Path, help="Path to yaml config.")
    return parser.parse_args()

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    f.close()
    return config

def extract_capture(config):
    CAPTURE_DIR = Path(config['capture_dir'])
    CROSS_VDB_DIR = Path(config['cross_vdb_dir'])
    map_ratio = float(config['map_ratio'])
    step = int(config['step'])
    num_query = int(config['num_query'])
    
    locations = sorted(os.listdir(CAPTURE_DIR))
    for location in locations:
        print(f"===Extract for {str.upper(location)}===")
        if location == "codabench":
            coda_dir = CAPTURE_DIR / "codabench"
            cross_coda_dir = CROSS_VDB_DIR / "codabench"
            os.makedirs(cross_coda_dir, exist_ok=True)
            
            files = [f for f in os.listdir(coda_dir) if f.endswith('.txt')]
            for file in files:
                file_path = coda_dir / file
                cp_file_path = cross_coda_dir / file
                
                shutil.copy(file_path, cp_file_path, follow_symlinks=True)
            print(f"Done for copy in new path: {cross_coda_dir}!")
            continue
        elif location in ['HYDRO', 'SUCCULENT']:
            location_dir = CAPTURE_DIR / location
            cross_location_dir = CROSS_VDB_DIR / location
            
            extract_location(location_dir,cross_location_dir, map_ratio, num_query, step)

def main():
    args = param_arg()
    config_path = args.config.resolve()
    config = load_config(config_path)
    extract_capture(config)
    
if __name__ == "__main__":
    main()
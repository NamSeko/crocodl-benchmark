import collections
import numpy as np
import re

class GetData:
    def __init__(self, 
                 sensor_path=None, 
                 trajectories_path=None, 
                 transform_path=None,
                 rig_path=None):
        self.sensor_path = sensor_path
        self.trajectories_path = trajectories_path
        self.transform_path = transform_path
        if not (self.sensor_path or self.trajectories_path or transform_path):
            raise EOFError("Must have path of sensor, trajectories and transform")
        self.rig_path = rig_path
        
    def get_data(self):
        Q_global, T_global = None, None
        if self.transform_path is not None:
            Q_global, T_global = self.get_transform()
        sensor_data = self.get_sensor()
        if self.rig_path is not None:
            rig_data = self.get_rig(sensor_data=sensor_data)
        data = collections.defaultdict()
        with open(self.trajectories_path, 'r') as f:
            for line in f:
                if not line.strip() or '#' in line:
                    continue
                parst = line.strip().split(', ')
                timestamp = parst[0]
                device_id = parst[1]
                quaterniont_matrix = np.array(parst[2:6], dtype=float)
                translate_matrix = np.array(parst[6:9], dtype=float)
                if self.rig_path is None:
                    data[timestamp] = {
                        'width': sensor_data[device_id]['width'],
                        'height': sensor_data[device_id]['height'],
                        'Q': quaterniont_matrix,
                        'T': translate_matrix,
                        'K': sensor_data[device_id]['K'],
                        'Q_global': Q_global,
                        'T_global': T_global
                    }
                else:
                    fl = rig_data.get(timestamp).get('fl')
                    fr = rig_data.get(timestamp).get('fr')
                    l = rig_data.get(timestamp).get('l')
                    r = rig_data.get(timestamp).get('r')
                    data[timestamp] = {
                        'Q': quaterniont_matrix,
                        'T': translate_matrix,
                        'Q_global': Q_global,
                        'T_global': T_global,
                        'fl': fl,
                        'fr': fr,
                        'l': l,
                        'r': r
                    }
        return data 
                     
    def get_sensor(self):
        sensor_data = collections.defaultdict()
        with open(self.sensor_path, 'r') as f:
            for line in f:
                if not line.strip() or '#' in line:
                    continue
                parst = line.strip().split(', ')
                sensor_id = parst[0]
                if len(parst) < 10 or 'zed2i' in sensor_id:
                    continue
                width, height = int(parst[4]), int(parst[5])
                fx, fy = float(parst[6]), float(parst[7])
                cx, cy = float(parst[8]), float(parst[9])
                intrinsic_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=float)
                sensor_data[sensor_id] = {
                    'width': width,
                    'height': height,
                    'K': intrinsic_matrix,
                }
        return sensor_data
    
    def get_rig(self, sensor_data):
        rig_data = collections.defaultdict(dict)
        with open(self.rig_path, 'r') as f:
            for line in f:
                if not line.strip() or '#' in line:
                    continue
                parts = line.strip().split(', ')
                sensor_id = parts[1]
                if len(parts) < 9 or 'zed2i' in sensor_id or 'imu' in sensor_id:
                    continue
                extract_timestamp = re.findall(r"\d+", parts[0].split('/')[-1])
                timestamp = extract_timestamp[0] if extract_timestamp else None
                quaterniont_matrix = np.array(parts[2:6], dtype=float)
                translation_matrix = np.array(parts[6:], dtype=float)
                data = {
                    'width': sensor_data[sensor_id]['width'],
                    'height': sensor_data[sensor_id]['height'],
                    'Q': quaterniont_matrix,
                    'T': translation_matrix,
                    'K': sensor_data[sensor_id]['K']
                }
                if 'camera-frontleft' in sensor_id or 'hetlf' in sensor_id:
                    rig_data[timestamp]['fl'] = data
                elif 'camera-frontright' in sensor_id or 'hetrf' in sensor_id:
                    rig_data[timestamp]['fr'] = data
                elif 'camera-left' in sensor_id or 'hetll' in sensor_id:
                    rig_data[timestamp]['l'] = data
                elif 'camera-right' in sensor_id or 'hetrr' in sensor_id:
                    rig_data[timestamp]['r'] = data
        return rig_data
    
    def get_transform(self):
        value = open(self.transform_path).read().splitlines()[1].strip().split(', ')
        quaterniont_matrix = np.array(value[1:5], dtype=float)
        translation_matrix = np.array(value[5:], dtype=float)
        return quaterniont_matrix, translation_matrix
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

class VisualizeCamPose:
    def __init__(self, color=(255, 255, 255), scale = 0.01, model = None):
        self.color = color
        self.scale = scale
        self.vis = None
        self.model = model
        if self.model == 'global':
            print("Global View")
        else:
            print("Local View")             
    def pose_matrix(self, Q, T):
        q_xyzw = np.array([Q[1], Q[2], Q[3], Q[0]])
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat(q_xyzw).as_matrix()
        pose[:3, 3] = np.array(T)
        return pose
    
    def check_ios(self, info):
        if 'width' in info.keys() and 'height' in info.keys():
            return True
        return False
        
    def plot_camera_frustum(self, K, width, height):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        corners = np.array([
            [(0 - cx) / fx, (0 - cy) / fy, 1],
            [(width - cx) / fx, (0 - cy) / fy, 1],
            [(width - cx) / fx, (height - cy) / fy, 1],
            [(0 - cx) / fx, (height - cy) / fy, 1]
        ]) * self.scale
        
        origin = np.zeros((1, 3))
        points = np.vstack([origin, corners])
        lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
        colors = [self.color for _ in lines]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    
    def visualize_robot_cams(self, rig_pose, cam_poses):
        rig_origin = rig_pose[:3, 3]
        cam_origins = [pose[:3, 3] for pose in cam_poses]

        points = [rig_origin] + cam_origins
        colors = [[1, 0, 0]] + [[0, 1, 0]] * len(cam_origins)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        lines = [[0, i] for i in range(1, len(points))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 1, 0]] * len(lines))
        self.vis.add_geometry(line_set)
        self.vis.add_geometry(pc)
    
    def visualize_ios(self, info):
        cam_pose = self.pose_matrix(Q=info['Q'], T=info['T'])
        if self.model == 'global':
            if info['Q_global'] is not None and info['T_global'] is not None:
                cam_pose_global = self.pose_matrix(Q=info['Q_global'], T=info['T_global'])
                cam_pose = cam_pose_global @ cam_pose  

        points = [cam_pose[:3, 3]]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector([[255, 0, 0]])
        self.vis.add_geometry(pc)
        
        frustum = self.plot_camera_frustum(
            K=info['K'], 
            width=info['width'], 
            height=info['height']
        )
        frustum.transform(cam_pose)
        self.vis.add_geometry(frustum)
    
    def visualize_hl_spot(self, info):
        cam_poses = []
        rig_pose = self.pose_matrix(info['Q'], info['T'])
        if self.model == 'global':
            if info['Q_global'] is not None and info['T_global'] is not None:
                cam_pose_global = self.pose_matrix(Q=info['Q_global'], T=info['T_global'])
                rig_pose = cam_pose_global @ rig_pose
        for cam_name, cam_info in info.items():
            if (not isinstance(cam_info, dict)) or ('K' not in cam_info):
                continue
            cam_pose_real = self.pose_matrix(cam_info['Q'], cam_info['T'])
            cam_pose_world = rig_pose @ cam_pose_real
            cam_poses.append(cam_pose_world)
            frustum = self.plot_camera_frustum(
                K=cam_info['K'], 
                width=cam_info['width'], 
                height=cam_info['width']
            )
            frustum.transform(cam_pose_world)
            self.vis.add_geometry(frustum)
        self.visualize_robot_cams(rig_pose=rig_pose, cam_poses=cam_poses)
    
    def visualization_single(self, info):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        if self.check_ios(info=info):
            self.visualize_ios(info=info)
        else:
            self.visualize_hl_spot(info=info)
        opt = self.vis.get_render_option()
        opt.line_width = 10
        opt.background_color = np.array([0, 0, 0])
        self.vis.run()
        self.vis.destroy_window()
        
    def visualization(self, many_infos):
        if len(many_infos) == 1:
            self.visualization_single(many_infos[0])
        elif isinstance(many_infos, dict):
            self.visualization_single(many_infos)
        else:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            for info in many_infos:
                if self.check_ios(info=info):
                    self.visualize_ios(info=info)
                else:
                    self.visualize_hl_spot(info=info)
            opt = self.vis.get_render_option()
            opt.line_width = 10
            opt.background_color = np.array([0, 0, 0])
            self.vis.run()
            self.vis.destroy_window()            
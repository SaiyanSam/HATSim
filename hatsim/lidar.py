"""Configurable ray-casting LiDAR, independent of navigation and policy."""
import math,numpy as np
from .geometry import collision_at

def scan(robot,world,azimuth_rays=None,elevation_angles=None,range_m=None,step=1.):
    spec=robot.spec.get('lidar',{})
    if not spec.get('enabled',False):return []
    n=int(azimuth_rays or spec.get('azimuth_rays',48)); angles=elevation_angles or spec.get('elevation_angles_deg',[-10,0,10]); radius=float(range_m or spec.get('range_m',30))
    mount=robot.spec.get('sensor_mount',{});offset=np.asarray(mount.get('position',[0,0,0]),float)
    cy=math.cos(robot.heading);sy=math.sin(robot.heading)
    origin=robot.position+np.array([cy*offset[0]-sy*offset[1],sy*offset[0]+cy*offset[1],offset[2]])
    # Mounting yaw and pitch rotate the angular scan pattern; roll reserved for future transforms.
    rpy=mount.get('rpy_deg',[0,0,0]);yaw=math.radians(rpy[2]);pitch=math.radians(rpy[1]);hits=[]
    for angle in np.linspace(0,2*math.pi,n,endpoint=False):
        for elev in angles:
            h=robot.heading+yaw+angle;v=math.radians(elev)+pitch
            ray=np.array([math.cos(v)*math.cos(h),math.cos(v)*math.sin(h),math.sin(v)])
            for distance in np.arange(step,radius+step*.5,step):
                p=origin+ray*distance
                if p[2]<=0 or collision_at(p,world['obstacles']):hits.append(p.tolist());break
    return hits

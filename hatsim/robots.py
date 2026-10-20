from dataclasses import dataclass,field
import math,numpy as np
from .navigation import astar,aircraft_guidance,wrap_angle
from .geometry import segment_collides
@dataclass
class Robot:
    id:str
    kind:str
    spec:dict
    position:np.ndarray
    heading:float=0.
    velocity:np.ndarray=field(default_factory=lambda:np.zeros(3))
    task_id:str|None=None
    alive:bool=True
    energy:float=1.
    path:list=field(default_factory=list)
    status:str='idle'
    distance:float=0.
    lidar_hits:int=0
    def state(self):
        return {'id':self.id,'kind':self.kind,'position':self.position.tolist(),'heading':float(self.heading),'velocity':self.velocity.tolist(),'energy':float(self.energy),'task_id':self.task_id,'alive':self.alive,'status':self.status,'distance':round(float(self.distance),4)}
    def plan(self,world,target):
        self.path=astar(world,self.position,target,self.kind)
        return bool(self.path)
    def advance(self,dt,world,target,others,service_radius=None):
        """Integrate bounded-speed waypoint motion and return whether near service target."""
        if not self.alive:return False
        if self.task_id is None:
            self.velocity[:]=0
            if self.kind=='fixed_wing':
                # Fixed-wing always flies holding/orbiting even without assigned work.
                self.heading+=math.radians(self.spec['turn_rate_deg'])*.5*dt
                v=self.spec['min_speed']; step=np.array([math.cos(self.heading)*v*dt,math.sin(self.heading)*v*dt,0])
                proposed=self.position+step
                if not segment_collides(self.position,proposed,world['obstacles'],self.spec['collision_radius']) and 1<proposed[0]<world['bounds'][0]-1 and 1<proposed[1]<world['bounds'][1]-1:
                    self.position=proposed;self.velocity=step/dt;self.distance+=np.linalg.norm(step)
                # Boundary navigation remains an explicit simplified hold pattern.
            return False
        if target is None:return False
        error=np.asarray(target)-self.position; dist=float(np.linalg.norm(error))
        if service_radius is None:
            service_radius=2.5 if self.kind=='ugv' else (7. if self.kind=='fixed_wing' else 3.5)
        if dist<service_radius and self.kind!='fixed_wing':
            self.velocity[:]=0;self.status='servicing';return True
        if not self.path and dist>=service_radius:
            self.plan(world,target)
        if self.path and np.linalg.norm(self.path[0]-self.position)<(2.5 if self.kind!='fixed_wing' else 5):self.path.pop(0)
        waypoint=self.path[0] if self.path else np.asarray(target)
        max_speed=float(self.spec['max_speed']); acceleration=float(self.spec['acceleration'])
        if self.kind=='fixed_wing':
            rate=math.radians(self.spec['turn_rate_deg']);self.heading=aircraft_guidance(self.position,self.heading,waypoint,rate,dt)
            vz=float(np.clip((waypoint[2]-self.position[2])*.7,-self.spec['climb_rate'],self.spec['climb_rate']))
            goal=np.array([math.cos(self.heading)*self.spec['cruise_speed'],math.sin(self.heading)*self.spec['cruise_speed'],vz])
            speed=max(self.spec['min_speed'],np.linalg.norm(self.velocity[:2]))
        else:
            diff=waypoint-self.position
            if self.kind=='ugv':diff[2]=0
            n=np.linalg.norm(diff)
            desired_dir=diff/n if n>1e-9 else np.zeros(3)
            goal=desired_dir*min(self.spec['cruise_speed'],max_speed)
            if self.kind=='multirotor':goal[2]=np.clip(goal[2],-self.spec['climb_rate'],self.spec['climb_rate'])
            # Simple predictive avoidance: repel from approaching robots.
            for other in others:
                if other.id==self.id or not other.alive:continue
                away=self.position-other.position;gap=np.linalg.norm(away)
                if .001<gap<self.spec['safety_radius']+other.spec['safety_radius']+2.:
                    goal+=away/gap * self.spec['cruise_speed'] * .8
            if self.kind=='ugv':goal[2]=0.
        dv=goal-self.velocity;n=np.linalg.norm(dv)
        if n>acceleration*dt:dv=dv/n*acceleration*dt
        self.velocity+=dv
        if self.kind=='fixed_wing':
            horizontal=self.velocity[:2]; norm=np.linalg.norm(horizontal)
            v=float(np.clip(norm,self.spec['min_speed'],max_speed))
            self.velocity[:2]=np.array([math.cos(self.heading),math.sin(self.heading)])*v
        else:
            norm=np.linalg.norm(self.velocity)
            if norm>max_speed:self.velocity*=max_speed/norm
        if self.kind=='ugv':self.velocity[2]=0.
        proposed=self.position+self.velocity*dt
        if self.kind=='ugv':proposed[2]=.6
        if self.kind=='fixed_wing':proposed[2]=max(5.,proposed[2])
        if self.kind=='multirotor':proposed[2]=max(1.2,proposed[2])
        if self.kind=='fixed_wing' and (proposed[0]<1 or proposed[0]>world['bounds'][0]-1 or proposed[1]<1 or proposed[1]>world['bounds'][1]-1):
            self.heading=wrap_angle(self.heading+math.radians(self.spec['turn_rate_deg'])*dt);self.velocity[:2]=np.array([math.cos(self.heading),math.sin(self.heading)])*self.spec['min_speed']
            proposed=np.clip(self.position+self.velocity*dt,[1,1,5],np.asarray(world['bounds'])-[1,1,1])
        proposed=np.clip(proposed,[1,1,.6],np.asarray(world['bounds'])-[1,1,1])
        # Predict collisions before moving. Ground robots and multirotors may brake;
        # aircraft must steer on a feasible forward arc instead of stopping.
        clearance=float(self.spec['collision_radius']) + float(world.get('navigation_margin_m',1.5))
        prediction_horizon=float(world.get('avoidance_lookahead_s',1.2))
        lookahead=proposed+self.velocity*prediction_horizon
        danger=segment_collides(self.position,lookahead,world['obstacles'],clearance)
        if danger:
            if self.kind=='fixed_wing':
                # Select a feasible bounded turn, favouring clear space ahead.
                max_turn=math.radians(self.spec['turn_rate_deg'])*dt
                for turn in (max_turn,-max_turn):
                    h=wrap_angle(self.heading+turn)
                    v=np.array([math.cos(h),math.sin(h),0.])*max(self.spec['min_speed'],np.linalg.norm(self.velocity[:2]))
                    v[2]=self.velocity[2]
                    test=self.position+v*(dt+prediction_horizon)
                    if (clearance<test[0]<world['bounds'][0]-clearance and clearance<test[1]<world['bounds'][1]-clearance and not segment_collides(self.position,test,world['obstacles'],clearance)):
                        self.heading=h;self.velocity=v;proposed=self.position+v*dt;break
                # If neither turn clears the obstacle, retain dynamics and let
                # the collision model determine the outcome.
            else:
                self.velocity[:]=0.
                proposed=self.position.copy()
                self.path=[] # Replan later from the current state.
                self.status='blocked'
                return False
        prev=self.position.copy();self.position=proposed;self.distance+=float(np.linalg.norm(proposed-prev))
        if np.linalg.norm(self.velocity[:2])>1e-5 and self.kind!='fixed_wing':
            aim=math.atan2(self.velocity[1],self.velocity[0]); self.heading+=np.clip(wrap_angle(aim-self.heading),-math.radians(self.spec['turn_rate_deg'])*dt,math.radians(self.spec['turn_rate_deg'])*dt)
        self.status='travelling'
        service=bool(np.linalg.norm(self.position-np.asarray(target))<service_radius)
        if self.kind=='fixed_wing' and service:
            # Service is gained only on a fly-by, never through loitering motionlessness.
            service=np.linalg.norm(self.velocity[:2])>=self.spec['min_speed']*.95
        if service:self.status='servicing'
        return service

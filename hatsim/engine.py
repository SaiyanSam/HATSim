import math
import numpy as np
from .config import platform_spec,world_spec,load_config
from .geometry import segment_collides,swept_spheres,collision_at
from .robots import Robot
from .tasks import Task,generate_tasks,transfer_work,load_task_sequence

class Simulator:
    def __init__(self,config=None,tasks=None):
        self.config=load_config() if config is None else dict(config)
        self.world=world_spec(self.config.get('world','industrial_campus'))
        self.rng=np.random.default_rng(np.random.SeedSequence([int(self.config.get('seed',42)),2027]))
        self.time=0.;self.ticks=0;self.events=[];self.robots=[]
        if tasks is not None:
            self.tasks=tasks
        else:
            self.tasks,seq_path,seq_mode=load_task_sequence(
                self.config.get('task_sequence',1),self.config,self.world,self.config.get('seed',42))
            self.config['task_sequence_file']=seq_path
            self.config['task_sequence_mode']=seq_mode
        self.task_index={t.id:t for t in self.tasks};self.lost_work=0.;self.collisions=0;self.done=False
        self._spawn()
    def emit(self,event,**info):
        self.events.append({'time':round(self.time,6),'event':event,**info})
    def _spawn(self):
        for kind,count in self.config.get('fleet',{'multirotor':2,'fixed_wing':1,'ugv':3}).items():
            spec=platform_spec(kind)
            for j in range(int(count)):
                z={'multirotor':12.,'fixed_wing':25.,'ugv':.6}[kind]
                for attempt in range(400):
                    p=np.array([float(self.rng.uniform(6,self.world['bounds'][0]-6)),float(self.rng.uniform(6,self.world['bounds'][1]-6)),z])
                    if not collision_at(p,self.world['obstacles'],spec['collision_radius']+1) and all(np.linalg.norm(p-r.position)>spec['safety_radius']+r.spec['safety_radius']+2 for r in self.robots):break
                else:raise RuntimeError('Cannot spawn safe fleet, reduce robot count')
                r=Robot('%s_%02d'%(kind,j+1),kind,spec,p,heading=float(self.rng.uniform(-math.pi,math.pi)))
                if kind=='fixed_wing':r.velocity=np.array([math.cos(r.heading)*spec['min_speed'],math.sin(r.heading)*spec['min_speed'],0.])
                self.robots.append(r)
    def release(self,robot,reason='release'):
        if robot.task_id is None:return
        t=self.task_index[robot.task_id]
        if t.owner==robot.id:
            t.owner=None
            if t.status not in ('completed','expired'):t.status='available'
        self.emit(reason,robot=robot.id,task=t.id,progress=round(t.progress,5))
        robot.task_id=None;robot.path=[];robot.status='idle'
    def claim(self,robot,task):
        if not robot.alive or task.status not in ('available','assigned','servicing') or task.owner not in (None,robot.id) or not task.eligible(robot.kind):return False
        if robot.task_id and robot.task_id!=task.id:self.release(robot,'switch_release')
        previous=self._robot_by_id(task.last_executor)
        old_kind=previous.kind if previous else None
        loss=transfer_work(task,robot.id,robot.kind,old_kind)
        if loss:self.lost_work+=loss;self.emit('progress_loss',task=task.id,amount=round(loss,5),old_executor=task.last_executor,new_executor=robot.id)
        if task.last_executor and task.last_executor!=robot.id:self.emit('handover',task=task.id,from_robot=task.last_executor,to_robot=robot.id)
        task.owner=robot.id;task.last_executor=robot.id;task.status='assigned';robot.task_id=task.id;robot.path=[]
        robot.plan(self.world,task.position)
        self.emit('claim',robot=robot.id,task=task.id)
        return True
    def _robot_by_id(self,id):return next((r for r in self.robots if r.id==id),None)
    def resolve_actions(self,actions):
        """Simultaneous claims, deterministic nearest-feasible-ETA wins, no central optimization."""
        for r in self.robots:
            if not r.alive:continue
            a=actions.get(r.id,'keep')
            if a=='release':self.release(r)
        candidates={}
        for r in self.robots:
            if not r.alive:continue
            a=actions.get(r.id,'keep')
            if isinstance(a,str) and a.startswith('claim:'):
                tid=a.split(':',1)[1];t=self.task_index.get(tid)
                if t and t.status not in ('unreleased','completed','expired') and t.eligible(r.kind) and t.owner in (None,r.id):
                    candidates.setdefault(tid,[]).append(r)
        for tid,claimers in sorted(candidates.items()):
            t=self.task_index[tid]
            winner=min(claimers,key=lambda r:(np.linalg.norm(r.position-np.asarray(t.position))/max(.01,r.spec['cruise_speed']),r.id))
            self.claim(winner,t)
            for loser in claimers:
                if loser.id!=winner.id:self.emit('claim_rejected',robot=loser.id,task=tid)
    def step(self,actions=None,dt=None):
        if self.done:return
        dt=float(dt or self.config.get('physics_dt',.1))
        self.time+=dt;self.ticks+=1
        for t in self.tasks:
            if t.status=='unreleased' and self.time+1e-8>=t.arrival:
                t.status='available';self.emit('task_arrival',task=t.id)
            if t.status not in ('unreleased','completed','expired') and self.time>=t.deadline-1e-9:
                if t.owner:
                    r=self._robot_by_id(t.owner)
                    if r:self.release(r,'deadline_release')
                t.status='expired';self.emit('task_expired',task=t.id,progress=round(t.progress,5))
        if actions:self.resolve_actions(actions)
        previous={r.id:r.position.copy() for r in self.robots}
        for r in self.robots:
            if not r.alive:continue
            target=self.task_index[r.task_id].position if r.task_id else None
            service_radius=None
            if r.task_id:
                task=self.task_index[r.task_id]
                regions=self.world.get('service_radii_m',{})
                service_radius=float(regions.get(task.type,{}).get(r.kind, 7. if r.kind=='fixed_wing' else (2.5 if r.kind=='ugv' else 3.5)))
            was_service=r.advance(dt,self.world,target,self.robots,service_radius=service_radius)
            r.energy-=dt/(r.spec['endurance_s'])*r.spec.get('energy_motion_multiplier',1.)
            if r.energy<=0:self.destroy(r,'energy_depleted');continue
            if was_service and r.task_id:
                t=self.task_index[r.task_id]
                if t.status not in ('completed','expired'):
                    t.status='servicing';t.progress=min(t.total_work,t.progress+t.rates[r.kind]*dt)
                    if t.progress>=t.total_work-1e-9:
                        t.completed_at=self.time;t.status='completed';self.emit('task_complete',task=t.id,robot=r.id,reward=t.reward)
                        self.release(r,'completion_release')
        for r in self.robots:
            if not r.alive:continue
            if segment_collides(previous[r.id],r.position,self.world['obstacles'],r.spec['collision_radius']):self.destroy(r,'environment_collision')
        for i,r in enumerate(self.robots):
            if not r.alive:continue
            for s in self.robots[i+1:]:
                if not s.alive:continue
                if swept_spheres(previous[r.id],r.position,previous[s.id],s.position,r.spec['collision_radius']+s.spec['collision_radius']):
                    self.collisions+=1;self.emit('robot_collision',robots=[r.id,s.id]);self.destroy(r,'robot_collision');self.destroy(s,'robot_collision')
        terminal=all(t.status in ('completed','expired') for t in self.tasks)
        hard_limit=float(self.config.get('max_runtime_s',300.0))
        if terminal or self.time>=hard_limit-1e-8:
            self.done=True
            self.emit('episode_end',reason='all_tasks_terminal' if terminal else 'safety_limit')
    def destroy(self,robot,reason):
        if not robot.alive:return
        self.release(robot,'destroyed_release');robot.alive=False;robot.velocity[:]=0;robot.status='destroyed';self.emit('robot_destroyed',robot=robot.id,reason=reason)
    def snapshot(self):
        return {'time':round(self.time,6),'robots':[r.state() for r in self.robots], 'tasks':[{'id':t.id,'status':t.status,'progress':round(t.progress,5),'owner':t.owner,'completed_at':t.completed_at} for t in self.tasks]}
    def metrics(self):
        done=[t for t in self.tasks if t.status=='completed' and t.completed_at<=t.deadline]
        return {'deadline_weighted_value':sum(t.reward for t in done),'completed':len(done),'total_tasks':len(self.tasks),'expired':sum(t.status=='expired' for t in self.tasks),'collisions':self.collisions,'destroyed':sum(not r.alive for r in self.robots),'lost_work':round(self.lost_work,4),'total_distance_m':round(sum(r.distance for r in self.robots),3),'handovers':sum(e['event']=='handover' for e in self.events),'time_s':round(self.time,3)}

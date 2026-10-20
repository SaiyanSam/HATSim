"""Optional PettingZoo ParallelEnv adapter. Install hatsim[marl]."""
import numpy as np
from .engine import Simulator
from .config import load_config

try:
    from pettingzoo import ParallelEnv
    from gymnasium import spaces
except ImportError:
    ParallelEnv=object;spaces=None

class HATSimParallelEnv(ParallelEnv):
    metadata={'name':'hatsim_v0','render_modes':[]}
    def __init__(self,config=None,max_candidates=12):
        if spaces is None:raise ImportError('Install with pip install -e ".[marl]"')
        self.config=load_config() if config is None else dict(config)
        self.max_candidates=max_candidates;self.sim=None;self.agents=[];self.possible_agents=[];self._last_value=0.;self._last_destroyed=0
        self._space=spaces.Box(low=-1e6,high=1e6,shape=(14+max_candidates*7,),dtype=np.float32)
        self._action_space=spaces.Discrete(max_candidates+2)
    def observation_space(self,agent):return self._space
    def action_space(self,agent):return self._action_space
    def state(self):
        """Global state for CTDE only. No unrealized future task data."""
        return {'time':self.sim.time,'robots':[r.state() for r in self.sim.robots], 'tasks':[{'id':t.id,'progress':t.progress,'owner':t.owner,'status':t.status,'reward':t.reward,'deadline':t.deadline} for t in self.sim.tasks if t.arrival<=self.sim.time]}
    def _candidates(self,robot):
        eligible=[t for t in self.sim.tasks if t.arrival<=self.sim.time and t.status in ('available','assigned','servicing') and t.eligible(robot.kind) and t.owner in (None,robot.id)]
        return sorted(eligible,key=lambda t:(t.deadline,t.id))[:self.max_candidates]
    def action_mask(self,agent):
        r=self.sim._robot_by_id(agent);mask=np.zeros(self.max_candidates+2,dtype=np.int8);mask[0]=1;mask[1]=int(r.task_id is not None)
        mask[2:2+len(self._candidates(r))]=1
        return mask
    def observe(self,agent):
        r=self.sim._robot_by_id(agent);work=0;deadline=0
        if r.task_id:
            t=self.sim.task_index[r.task_id];work=t.progress/max(t.total_work,1);deadline=t.deadline-self.sim.time
        own=np.array([*r.position,*r.velocity,r.energy,r.heading,float(r.alive),float(r.kind=='multirotor'),float(r.kind=='fixed_wing'),float(r.kind=='ugv'),work,deadline],dtype=np.float32)
        candidates=[]
        for t in self._candidates(r):candidates.extend([*(np.array(t.position)-r.position),t.reward,t.deadline-self.sim.time,t.progress/t.total_work,t.rates[r.kind]])
        return np.concatenate([own,np.array((candidates+[0.]*(self.max_candidates*7-len(candidates))),dtype=np.float32)])
    def reset(self,seed=None,options=None):
        config=dict(self.config)
        if seed is not None:config['seed']=seed
        self.sim=Simulator(config);self.agents=[r.id for r in self.sim.robots];self.possible_agents=list(self.agents);self._last_value=0;self._last_destroyed=0
        return {a:self.observe(a) for a in self.agents},{a:{'action_mask':self.action_mask(a)} for a in self.agents}
    def step(self,actions):
        decisions={}
        for agent in self.agents:
            r=self.sim._robot_by_id(agent);act=int(actions.get(agent,0))
            if act==1:decisions[agent]='release'
            elif act>=2:
                cand=self._candidates(r)
                if act-2<len(cand):decisions[agent]='claim:'+cand[act-2].id
        self.sim.resolve_actions(decisions)
        dt=float(self.config.get('physics_dt',.1));horizon=float(self.config.get('allocation_period_s',2.))
        for _ in range(max(1,round(horizon/dt))):
            if self.sim.done:break
            self.sim.step(dt=dt)
        new=self.sim.metrics();gain=float(new['deadline_weighted_value'])-self._last_value;collision_delta=int(new['destroyed'])-self._last_destroyed
        self._last_value=float(new['deadline_weighted_value']);self._last_destroyed=int(new['destroyed'])
        current=list(self.agents);obs={};rew={};terms={};trunc={};info={}
        for a in current:
            robot=self.sim._robot_by_id(a);terms[a]=not robot.alive;trunc[a]=bool(self.sim.done and robot.alive)
            rew[a]=gain-30.*collision_delta;obs[a]=self.observe(a);info[a]={'action_mask':self.action_mask(a),'elapsed_time':horizon}
        self.agents=[a for a in current if not terms[a] and not trunc[a]]
        return obs,rew,terms,trunc,info
    def close(self):pass

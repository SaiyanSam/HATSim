import json
import numpy as np
from hatsim.config import load_config
from hatsim.engine import Simulator
from hatsim.tasks import Task,transfer_work
from hatsim.records import load_run
from hatsim.cli import run
from hatsim.geometry import swept_spheres

def conf(**updates):
    x=load_config();x.update({'duration_s':5,'initial_tasks':2,'arrival_rate_per_min':0,'fleet':{'multirotor':1,'fixed_wing':1,'ugv':1}});x.update(updates);return x

def test_spawn():
    s=Simulator(conf());assert len(s.robots)==3
    assert {r.kind for r in s.robots}=={'multirotor','fixed_wing','ugv'}

def test_completes_service():
    s=Simulator(conf(fleet={'multirotor':1,'fixed_wing':0,'ugv':0}))
    r=s.robots[0];t=Task('X','general_inspection',r.position.tolist(),0,100,100,10,{'multirotor':10,'fixed_wing':0,'ugv':0})
    s.tasks=[t];s.task_index={'X':t};s.step({'%s'%r.id:'claim:X'},.1)
    for _ in range(15):s.step(dt=.1)
    assert t.status=='completed';assert s.metrics()['completed']==1

def test_handover_retention():
    s=Simulator(conf(fleet={'multirotor':1,'fixed_wing':0,'ugv':1}))
    mr=next(r for r in s.robots if r.kind=='multirotor');ugv=next(r for r in s.robots if r.kind=='ugv')
    t=Task('X','general_inspection',[20,20,.6],0,100,100,100,{'multirotor':10,'fixed_wing':0,'ugv':5},progress=80,status='available')
    s.tasks=[t];s.task_index={'X':t};s.claim(mr,t);s.release(mr)
    s.claim(ugv,t)
    assert abs(t.progress-48)<1e-9
    assert abs(s.lost_work-32)<1e-9
    assert any(e['event']=='handover' for e in s.events)

def test_no_double_retention():
    s=Simulator(conf(fleet={'multirotor':1,'fixed_wing':0,'ugv':1}))
    a,b=s.robots;t=Task('X','general_inspection',[12,12,.6],0,100,100,100,{'multirotor':10,'ugv':5},progress=50,status='available')
    s.tasks=[t];s.task_index={'X':t};s.claim(a,t);s.release(a);s.claim(a,t)
    assert t.progress==50

def test_collision_detection():
    assert swept_spheres([0,0,0],[10,0,0],[5,0,0],[5,0,0],1.0)
    assert not swept_spheres([0,0,0],[1,0,0],[5,0,0],[5,1,0],1.0)

def test_numbering_and_replay(tmp_path):
    folder1,sim=run(conf(duration_s=1),tmp_path,False)
    folder2,_=run(conf(duration_s=1),tmp_path,False)
    assert folder1.name=='0001' and folder2.name=='0002'
    one=load_run(folder1);assert len(one['frames'])>1
    assert one['frames'][-1]['robots'][0]['position']==sim.snapshot()['robots'][0]['position']
    assert (folder1/'replay'/'trajectory.npz').exists()

def test_fixed_wing_forward_motion():
    s=Simulator(conf(fleet={'multirotor':0,'fixed_wing':1,'ugv':0}));r=s.robots[0]
    before=r.position.copy();s.step(dt=.1)
    assert np.linalg.norm(r.position-before)>0
    assert np.linalg.norm(r.velocity[:2])>=r.spec['min_speed']-.1

def test_deterministic_trajectories():
    a=Simulator(conf(seed=123));b=Simulator(conf(seed=123))
    for _ in range(10):
        a.step(dt=.1);b.step(dt=.1)
    assert a.snapshot()==b.snapshot()

def test_task_stream_log_is_initial_state(tmp_path):
    folder,_=run(conf(duration_s=1),tmp_path,False)
    import json
    data=json.loads((folder/'task_arrivals.jsonl').read_text().splitlines()[0])
    assert data['progress']==0 and data['status']=='unreleased'

import json,os,time
from pathlib import Path
import numpy as np,yaml

def allocate(root='logs'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # OS atomic exclusive mkdir. Concurrent processes recheck on conflict.
    for _ in range(1000):
        nums=[int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        idx=max(nums,default=0)+1
        path=root/f'{idx:04d}'
        try:path.mkdir();return path
        except FileExistsError:time.sleep(.005)
    raise RuntimeError('Could not allocate run directory')

def save_run(sim,snapshots,root='logs',policy='highest_reward'):
    folder=allocate(root)
    with open(folder/'manifest.yaml','w') as f:yaml.safe_dump({'version':'0.1.0','config':sim.config,'policy':policy,'world':sim.world['name'],'number_of_frames':len(snapshots),'fleet':{k:sum(r.kind==k for r in sim.robots) for k in ('multirotor','fixed_wing','ugv')}},f)
    with open(folder/'world_snapshot.yaml','w') as f:yaml.safe_dump(sim.world,f)
    with open(folder/'platform_specs.yaml','w') as f:yaml.safe_dump({r.kind:r.spec for r in sim.robots},f)
    with open(folder/'task_arrivals.jsonl','w') as f:
        for t in sim.tasks:
            item=t.to_dict();item.update(progress=0.0,owner=None,last_executor=None,status='unreleased',completed_at=None)
            f.write(json.dumps(item)+'\n')
    with open(folder/'events.jsonl','w') as f:
        for e in sim.events:f.write(json.dumps(e)+'\n')
    with open(folder/'metrics.json','w') as f:json.dump(sim.metrics(),f,indent=2)
    # Full-state JSONL is the canonical source for exact seek/replay.
    with open(folder/'frames.jsonl','w') as f:
        for snap in snapshots:f.write(json.dumps(snap)+'\n')
    # Compressed trajectory snapshot for efficient external tooling.
    (folder/'replay').mkdir()
    np.savez_compressed(folder/'replay'/'trajectory.npz',times=np.array([s['time'] for s in snapshots]),positions=np.array([[r['position'] for r in s['robots']] for s in snapshots]),headings=np.array([[r['heading'] for r in s['robots']] for s in snapshots]),alive=np.array([[r['alive'] for r in s['robots']] for s in snapshots]),robot_ids=np.array([r.id for r in sim.robots]))
    return folder

def load_run(folder):
    folder=Path(folder)
    with open(folder/'manifest.yaml') as f:manifest=yaml.safe_load(f)
    with open(folder/'world_snapshot.yaml') as f:world=yaml.safe_load(f)
    with open(folder/'task_arrivals.jsonl') as f:tasks=[json.loads(s) for s in f if s.strip()]
    with open(folder/'frames.jsonl') as f:frames=[json.loads(s) for s in f if s.strip()]
    with open(folder/'events.jsonl') as f:events=[json.loads(s) for s in f if s.strip()]
    return {'manifest':manifest,'world':world,'tasks':tasks,'frames':frames,'events':events}

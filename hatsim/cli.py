import argparse,json
from pathlib import Path
from .config import load_config
from .engine import Simulator
from .policies import select_actions,available_policies
from .records import save_run

def run(config,log_root='logs',verbose=True):
    sim=Simulator(config);frames=[sim.snapshot()]
    dt=float(config.get('physics_dt',.1));allocation=float(config.get('allocation_period_s',2.));record=float(config.get('record_dt',.2))
    next_alloc=0.;next_record=record
    while not sim.done:
        act=select_actions(sim,config.get('policy','highest_reward')) if sim.time+1e-8>=next_alloc else None
        if act is not None:next_alloc+=allocation
        sim.step(act,dt)
        if sim.time+1e-8>=next_record or sim.done:
            frames.append(sim.snapshot());next_record+=record
    folder=save_run(sim,frames,log_root,config.get('policy','highest_reward'))
    if verbose:print(json.dumps({'log_directory':str(folder),'metrics':sim.metrics()},indent=2))
    return folder,sim

def main():
    p=argparse.ArgumentParser(description='HATSim headless simulator')
    p.add_argument('--config');p.add_argument('--world');p.add_argument('--seed',type=int);p.add_argument('--duration',type=float)
    p.add_argument('--task-sequence', help='Task sequence ID or YAML path')
    p.add_argument('--timeout', type=float, help='Total episode timeout (default 300 s)')
    p.add_argument('--arrival-window', type=float, help='Window for new task arrivals in seconds')
    p.add_argument('--multirotors',type=int);p.add_argument('--fixed-wings',type=int);p.add_argument('--ugvs',type=int)
    p.add_argument('--policy',help='Policy folder name under policies/');p.add_argument('--list-policies',action='store_true',help='List installed policy modules and exit');p.add_argument('--log-root',default='logs')
    p.add_argument('--replay',help='Visualize existing numbered log folder (no physics rerun)')
    p.add_argument('--gui',action='store_true',help='Open interactive PyVista/Qt GUI if optional dependencies exist')
    p.add_argument('--headless',action='store_true')
    args=p.parse_args()
    if args.list_policies:
        print('\n'.join(available_policies()))
        return
    if args.replay:
        from .viewer import replay
        replay(args.replay,prefer_qt=args.gui)
        return
    c=load_config(args.config)
    if args.world:c['world']=args.world
    if args.task_sequence is not None:c['task_sequence']=int(args.task_sequence) if args.task_sequence.isdigit() else args.task_sequence
    if args.timeout is not None:c['max_runtime_s']=args.timeout
    if args.seed is not None:c['seed']=args.seed
    if args.duration is not None:
        c['duration_s']=args.duration
        if args.arrival_window is None:c['arrival_window_s']=args.duration
    if args.arrival_window is not None:c['arrival_window_s']=args.arrival_window
    if args.policy:c['policy']=args.policy
    for key,arg in [('multirotor',args.multirotors),('fixed_wing',args.fixed_wings),('ugv',args.ugvs)]:
        if arg is not None:c.setdefault('fleet',{})[key]=arg
    folder,_=run(c,args.log_root)
    if args.gui:
        from .viewer import replay
        replay(folder,prefer_qt=True)
if __name__=='__main__':main()

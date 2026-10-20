from dataclasses import dataclass,field
import numpy as np
TYPES=['point_inspection','corridor_survey','general_inspection']
RATES={
 'point_inspection':{'multirotor':8.,'fixed_wing':0.,'ugv':3.},
 'corridor_survey':{'multirotor':5.,'fixed_wing':12.,'ugv':0.},
 'general_inspection':{'multirotor':7.,'fixed_wing':4.,'ugv':5.},
}
RETENTION={'multirotor':{'multirotor':1.,'fixed_wing':.8,'ugv':.6},'fixed_wing':{'multirotor':.9,'fixed_wing':1.,'ugv':.5},'ugv':{'multirotor':.7,'fixed_wing':.5,'ugv':1.}}
@dataclass
class Task:
    id:str
    type:str
    position:list
    arrival:float
    deadline:float
    reward:float
    total_work:float
    rates:dict
    retention:dict=field(default_factory=lambda:RETENTION)
    progress:float=0.
    owner:str|None=None
    last_executor:str|None=None
    status:str='unreleased'
    completed_at:float|None=None
    def to_dict(self):
        return {k:getattr(self,k) for k in self.__dataclass_fields__}
    def eligible(self,kind):return self.rates.get(kind,0)>0

def generate_tasks(rng,world,n_initial,arrival_rate,duration,max_tasks):
    tasks=[]; t=0.
    while len(tasks)<max_tasks:
        if len(tasks)>=n_initial:
            if arrival_rate<=0:break
            t+=float(rng.exponential(60/arrival_rate))
            if t>duration:break
        kind=TYPES[int(rng.integers(0,len(TYPES)))]; rates=RATES[kind].copy()
        # Keep cross-platform inspect tasks visible in each environment.
        for _ in range(200):
            x=float(rng.uniform(8,world['bounds'][0]-8)); y=float(rng.uniform(8,world['bounds'][1]-8))
            z=0.6 if kind=='point_inspection' and rng.random()<.45 else float(rng.uniform(8,min(20,world['bounds'][2]-4)))
            pos=[x,y,z]
            from .geometry import collision_at
            if not collision_at(pos,world['obstacles'],radius=4):break
        else:continue
        if kind=='general_inspection' and z>3:
            rates['ugv']=0. # UGV cannot service elevated regions
        if kind=='point_inspection' and z>3:rates['ugv']=0.
        total=float(rng.choice([40,60,100])); deadline=t+float(rng.uniform(90,230))*float(world.get('deadline_scale',1.0))
        tasks.append(Task('T%04d'%(len(tasks)+1),kind,pos,round(t,5),round(deadline,5),float(rng.choice([50,100,150])),total,rates))
    return tasks

def transfer_work(task,new_executor,new_kind,old_kind):
    if old_kind is not None and old_kind!=new_kind:
        factor=task.retention.get(old_kind,{}).get(new_kind,0.)
        prior=task.progress;task.progress*=factor
        return prior-task.progress
    return 0.


def load_task_sequence(sequence, config, world, seed):
    """Build task realization independently of robot spawning and policy RNG.

    Mode=generated uses a seeded workload generator. Mode=explicit loads
    an exact task stream supplied by the user (no random generation).
    """
    from pathlib import Path
    import yaml
    from .config import CONFIGS
    name = f"seq_{int(sequence):03d}.yaml" if str(sequence).isdigit() else str(sequence)
    path = Path(name)
    if not path.is_file():
        path = CONFIGS / "task_sequences" / name
    if not path.is_file():
        raise FileNotFoundError(f"Task sequence not found: {name}")
    with open(path, encoding="utf8") as f:
        definition = yaml.safe_load(f) or {}
    mode = definition.get("mode", "generated")
    if mode == "explicit":
        output = []
        for entry in definition.get("tasks", []):
            item = dict(entry)
            if "arrival_time" in item:
                item["arrival"] = item.pop("arrival_time")
            if "service_rates" in item:
                item["rates"] = item.pop("service_rates")
            output.append(Task(**item))
        if len({t.id for t in output}) != len(output):
            raise ValueError("Task IDs must be unique")
        return output, str(path), mode
    if mode != "generated":
        raise ValueError(f"Unknown task sequence mode: {mode}")
    # Use separate RNG from the fleet spawning RNG. Never include policy
    # or fleet composition in this seed, allowing paired comparisons.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(sequence) if str(sequence).isdigit() else 1, 1031]))
    # Config values override Sequence 1 defaults for backwards compatibility
    # with existing CLI arguments and scripts. For new sequence IDs the YAML
    # determines their distribution; arrival_window_s remains CLI-configurable.
    primary = str(sequence) in ("1", "001")
    def parameter(key, fallback):
        return config.get(key, definition.get(key, fallback)) if primary else definition.get(key, config.get(key, fallback))
    tasks = generate_tasks(
        rng, world, int(parameter("initial_tasks", 2)),
        float(parameter("arrival_rate_per_min", 4.0)),
        float(config.get("arrival_window_s", definition.get("arrival_window_s", 60.0))),
        int(parameter("max_tasks", 40)),
    )
    return tasks, str(path), mode

"""A* grids for ground/air, heading-constrained local tracking for aircraft."""
import heapq, math
import numpy as np
from .geometry import collision_at,segment_collides

def astar(world,start,goal,kind='multirotor',resolution=5.0):
    bound=np.array(world['bounds'],float)
    dim=2 if kind=='ugv' else 3
    res=resolution if kind!='ugv' else 3.0
    limits=np.maximum(1,np.floor(bound[:dim]/res).astype(int))
    def quant(p):return tuple(np.clip(np.round(np.asarray(p)[:dim]/res).astype(int),0,limits))
    source=quant(start); target=quant(goal)
    if source==target:return [np.asarray(goal,float)]
    offsets=[tuple(int(i==j)*sgn for i in range(dim)) for j in range(dim) for sgn in [-1,1]]
    # Permit diagonal steps in each platform's grid.
    import itertools
    offsets=[d for d in itertools.product((-1,0,1),repeat=dim) if any(d)]
    def pos(n):return np.array((n[0]*res,n[1]*res,0.6 if dim==2 else n[2]*res),float)
    obstacle_margin=float(world.get('navigation_margin_m',1.5))
    def free(n):
        if any(v<0 or v>limits[i] for i,v in enumerate(n)):return False
        p=pos(n)
        return not collision_at(p,world['obstacles'],radius=(0.85 if kind=='ugv' else 1.5)+obstacle_margin)
    if not free(target):
        # Goal is invalid. Caller should handle unreachable task instead of driving through walls.
        return []
    opened=[(0,source)]; g={source:0}; parent={}; visited=set(); iters=0
    while opened and iters<65000:
        iters+=1; _,n=heapq.heappop(opened)
        if n in visited:continue
        visited.add(n)
        if n==target:
            path=[n]
            while n in parent:
                n=parent[n];path.append(n)
            path=list(reversed(path))
            out=[pos(q) for q in path[1:]]
            out.append(np.asarray(goal,float))
            return out
        for delta in offsets:
            m=tuple(n[i]+delta[i] for i in range(dim))
            if not free(m):continue
            if segment_collides(pos(n),pos(m),world['obstacles'],(0.85 if kind=='ugv' else 1.5)+obstacle_margin):continue
            score=g[n]+np.linalg.norm(np.asarray(delta))*res
            if score<g.get(m,float('inf')):
                parent[m]=n;g[m]=score
                heapq.heappush(opened,(score+np.linalg.norm(np.array(m)-np.array(target))*res,m))
    return []

def wrap_angle(a):return (a+math.pi)%(2*math.pi)-math.pi

def aircraft_guidance(position,heading,target,max_turn_rate,dt):
    """Bounded-curvature pursuit: aircraft never instantly reverses."""
    delta=np.asarray(target)-np.asarray(position); desired=math.atan2(delta[1],delta[0]); error=wrap_angle(desired-heading)
    return heading+np.clip(error,-max_turn_rate*dt,max_turn_rate*dt)

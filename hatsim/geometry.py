import numpy as np

def inside_box(p,box,pad=0):
    p=np.asarray(p); c=np.asarray(box['center']); half=np.asarray(box['size'])/2 + pad
    return bool(np.all(np.abs(p-c)<=half))

def collision_at(p, obstacles, radius=0):
    return any(inside_box(p,o,radius) for o in obstacles)

def segment_box(a,b,box,pad=0):
    """Slab intersection for swept point against inflated axis-aligned box."""
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    c=np.asarray(box['center']); h=np.asarray(box['size'])/2+pad
    delta=b-a; lo=0.; hi=1.
    for v in range(3):
        if abs(delta[v])<1e-12:
            if a[v]<c[v]-h[v] or a[v]>c[v]+h[v]: return False
        else:
            t0=(c[v]-h[v]-a[v])/delta[v]; t1=(c[v]+h[v]-a[v])/delta[v]
            lo=max(lo,min(t0,t1)); hi=min(hi,max(t0,t1))
            if lo>hi: return False
    return True

def segment_collides(a,b,obstacles,pad=0):
    return any(segment_box(a,b,o,pad) for o in obstacles)

def swept_spheres(a,b,c,d,radius):
    """Continuous relative-motion collision in one simulation step."""
    u=np.asarray(a)-np.asarray(c); v=(np.asarray(b)-np.asarray(a))-(np.asarray(d)-np.asarray(c))
    q=np.clip(-float(np.dot(u,v))/(float(np.dot(v,v))+1e-14),0,1)
    return np.linalg.norm(u+q*v) <=radius

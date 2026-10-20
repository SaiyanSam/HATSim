"""Offline trajectory viewer: Matplotlib fallback; optional PyVistaQt desktop GUI."""
import json
from pathlib import Path
import numpy as np
from .records import load_run
COLORS={'multirotor':'#0db7a5','fixed_wing':'#ec9a28','ugv':'#4386dc'}

def task_remaining_scale(progress, total_work):
    """Fraction of initial task-marker radius remaining (0 at completion)."""
    if total_work <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - float(progress) / float(total_work)))

def replay(folder,prefer_qt=False):
    data=load_run(folder)
    if prefer_qt:
        try:return qt_replay(data)
        except ImportError:
            print('Optional PyVista/PySide6/PyVistaQt not installed: using Matplotlib 3D viewer.')
    return matplotlib_replay(data)

def _build_axes(ax,world):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for o in world['obstacles']:
        c=np.array(o['center']);h=np.array(o['size'])/2
        verts=np.array([[c[0]+x*h[0],c[1]+y*h[1],c[2]+z*h[2]] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        faces=[[verts[k] for k in indices] for indices in [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]]
        ax.add_collection3d(Poly3DCollection(faces,alpha=.18,facecolor='slategray',edgecolor='#637487'))
    ax.set(xlim=(0,world['bounds'][0]),ylim=(0,world['bounds'][1]),zlim=(0,world['bounds'][2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')

def matplotlib_replay(data):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button,Slider,RadioButtons
    frames=data['frames'];world=data['world'];fig=plt.figure(figsize=(12,8));ax=fig.add_axes([.04,.19,.75,.75],projection='3d')
    _build_axes(ax,world)
    state={'index':0,'playing':False,'speed':1.,'view':'3D','follow':None}
    robot_ids=[r['id'] for r in frames[0]['robots']]
    trajectories={rid:np.array([next(r['position'] for r in frame['robots'] if r['id']==rid) for frame in frames]) for rid in robot_ids}
    points={};trails={};task_artists={}
    for r in frames[0]['robots']:
        color=COLORS[r['kind']];points[r['id']]=ax.plot([],[],[],marker='o' if r['kind']!='fixed_wing' else '^',linestyle='',color=color,markersize=8,label=r['id'])[0]
        trails[r['id']]=ax.plot([],[],[],color=color,alpha=.3,linewidth=1)[0]
    for task in data['tasks']:
        xyz=task['position'];task_artists[task['id']]=ax.plot([xyz[0]],[xyz[1]],[xyz[2]],'o',color='gold',alpha=0,markersize=6)[0]
    ax.legend(loc='upper left',fontsize=7)
    title=ax.set_title('')
    def draw(index):
        index=max(0,min(index,len(frames)-1));state['index']=index;frame=frames[index]
        for r in frame['robots']:
            xyz=np.array(r['position']);artist=points[r['id']];artist.set_data([xyz[0]],[xyz[1]]);artist.set_3d_properties([xyz[2]]);artist.set_alpha(1 if r['alive'] else .15)
            v=trajectories[r['id']][:index+1];trails[r['id']].set_data(v[:,0],v[:,1]);trails[r['id']].set_3d_properties(v[:,2])
        st={t['id']:t for t in frame['tasks']}
        for t in data['tasks']:
            artist=task_artists[t['id']];s=st[t['id']]['status']
            spec=t
            remaining=task_remaining_scale(st[t['id']].get('progress', 0), float(spec.get('total_work', 1)))
            artist.set_alpha(0 if s in ('unreleased', 'expired', 'completed') or remaining <= 0 else .9)
            artist.set_markersize(6 * remaining)
            artist.set_color('gold')
        if state['view']=='Top':ax.view_init(elev=90,azim=-90)
        elif state['view']=='Side':ax.view_init(elev=5,azim=-90)
        elif state['view']=='3D':ax.view_init(elev=32,azim=-55)
        elif state['view']=='Follow':
            r=frame['robots'][0];p=r['position'];ax.set(xlim=(p[0]-22,p[0]+22),ylim=(p[1]-22,p[1]+22));ax.view_init(elev=30,azim=-55)
        title.set_text(f"HATSim run {data['manifest']['world']} | t={frame['time']:.1f}s | frame {index+1}/{len(frames)}")
        if slider.val!=index:slider.set_val(index)
        fig.canvas.draw_idle()
    slider_ax=fig.add_axes([.08,.10,.66,.03]);slider=Slider(slider_ax,'Frame',0,len(frames)-1,valinit=0,valstep=1)
    slider.on_changed(lambda value:draw(int(value)) if int(value)!=state['index'] else None)
    buttons=[]
    for i,(label,action) in enumerate([('Play/Pause',lambda:state.update(playing=not state['playing'])),('Top',lambda:set_view('Top')),('3D',lambda:set_view('3D')),('Side',lambda:set_view('Side')),('Follow',lambda:set_view('Follow')),('Free',lambda:set_view('Free'))]):
        a=fig.add_axes([.02+i*.126,.025,.118,.045]);b=Button(a,label);b.on_clicked(lambda evt,fn=action:fn());buttons.append(b)
    def set_view(view):
        state['view']=view
        if view!='Follow':ax.set(xlim=(0,world['bounds'][0]),ylim=(0,world['bounds'][1]))
        draw(state['index'])
    def tick():
        if state['playing']:
            draw(state['index']+1)
            if state['index']>=len(frames)-1:state['playing']=False
    timer=fig.canvas.new_timer(interval=40);timer.add_callback(tick);timer.start()
    draw(0);plt.show()

def _vehicle_mesh(pv, kind):
    """Small local-coordinate meshes; +X is the vehicle's forward direction."""
    if kind == 'multirotor':
        body = pv.Box(bounds=(-0.75, 0.75, -0.35, 0.35, -0.18, 0.18))
        arm_x = pv.Box(bounds=(-0.12, 0.12, -1.35, 1.35, -0.08, 0.08))
        arm_y = pv.Box(bounds=(-1.35, 1.35, -0.12, 0.12, -0.08, 0.08))
        mesh = body.merge(arm_x).merge(arm_y)
        for x in (-1.2, 1.2):
            for y in (-1.2, 1.2):
                mesh = mesh.merge(pv.Cylinder(center=(x, y, 0.10), direction=(0,0,1), radius=0.35, height=0.08, resolution=12))
        return mesh
    if kind == 'fixed_wing':
        fuselage = pv.Box(bounds=(-1.65, 1.65, -0.18, 0.18, -0.13, 0.13))
        wing = pv.Box(bounds=(-0.15, 0.47, -2.1, 2.1, -0.06, 0.06))
        tail = pv.Box(bounds=(-1.45, -1.05, -0.9, 0.9, -0.05, 0.05))
        fin = pv.Box(bounds=(-1.52, -1.15, -0.06, 0.06, 0.02, 0.55))
        nose = pv.Cone(center=(1.85,0,0), direction=(1,0,0), radius=0.19, height=0.48, resolution=12)
        return fuselage.merge(wing).merge(tail).merge(fin).merge(nose)
    chassis = pv.Box(bounds=(-0.9, 0.9, -0.62, 0.62, -0.16, 0.33))
    roof = pv.Box(bounds=(-0.45,0.5,-0.38,0.38,0.33,0.62))
    mesh = chassis.merge(roof)
    for x in (-0.61, 0.61):
        for y in (-0.69, 0.69):
            wheel = pv.Cylinder(center=(x,y,-0.20), direction=(0,1,0), radius=0.31, height=0.20, resolution=14)
            mesh = mesh.merge(wheel)
    return mesh


def qt_replay(data):
    """Offline PyVista/Qt replay; presentation changes do not alter simulation data."""
    import math
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
        QHBoxLayout, QPushButton, QSlider, QLabel, QScrollArea, QProgressBar,
        QFrame, QSizePolicy)
    from PySide6.QtCore import Qt, QTimer
    from pyvistaqt import QtInteractor
    import pyvista as pv

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setWindowTitle('HATSim — Trajectory Replay')
    root = QWidget()
    vertical = QVBoxLayout(root)
    vertical.setContentsMargins(8, 8, 8, 8)
    vertical.setSpacing(7)
    frames = data['frames']
    world = data['world']
    bounds = world['bounds']
    task_specs = {t['id']: t for t in data['tasks']}

    # Persistent legend at the top (no moving/in-world labels required).
    legend = QHBoxLayout()
    heading = QLabel('HATSim  |  Recorded trajectory replay')
    heading.setStyleSheet('font-weight: bold; font-size: 15px;')
    legend.addWidget(heading)
    legend.addStretch()
    platform_legend = {}
    for kind, caption in [('multirotor', '✚  Multirotor'),
                          ('fixed_wing', '✈  Fixed-wing'),
                          ('ugv', '■  UGV')]:
        label = QLabel(caption)
        label.setStyleSheet(f'color: {COLORS[kind]}; font-weight: 600; padding: 4px;')
        legend.addWidget(label)
        platform_legend[kind] = (label, caption)
    for caption, color in [('●  Task', '#eabf3a'),
                           ('▰  Building', '#8292a2')]:
        label = QLabel(caption)
        label.setStyleSheet(f'color: {color}; font-weight: 600; padding: 4px;')
        legend.addWidget(label)
    vertical.addLayout(legend)

    middle = QHBoxLayout()
    middle.setSpacing(8)
    vertical.addLayout(middle, stretch=1)
    plot_container = QWidget()
    viewport_layout = QVBoxLayout(plot_container)
    viewport_layout.setContentsMargins(0, 0, 0, 0)
    plot = QtInteractor(plot_container)
    viewport_layout.addWidget(plot.interactor)
    middle.addWidget(plot_container, stretch=4)
    plot.set_background('#172c3c')

    # The ground extends across the whole map (z=0), not just beneath buildings.
    ground = pv.Plane(center=(bounds[0]/2, bounds[1]/2, -0.04),
                      direction=(0,0,1), i_size=bounds[0], j_size=bounds[1],
                      i_resolution=1, j_resolution=1)
    plot.add_mesh(ground, color='#526c5d', opacity=1, name='ground_plane',
                  lighting=True)
    plot.add_mesh(pv.Box(bounds=[0,bounds[0],0,bounds[1],-.07,-.05]),
                  color='#31493e', name='ground_edge')
    for o in world['obstacles']:
        c = o['center']; h = [v/2 for v in o['size']]
        actor_mesh = pv.Box(bounds=[c[0]-h[0], c[0]+h[0],
                                    c[1]-h[1], c[1]+h[1],
                                    c[2]-h[2], c[2]+h[2]])
        plot.add_mesh(actor_mesh, color='#8292a2', opacity=.85, show_edges=True,
                      edge_color='#526171')

    # Meshes are oriented from the recorded yaw, rather than display-only dots.
    robots = {}
    for r in frames[0]['robots']:
        actor = plot.add_mesh(_vehicle_mesh(pv, r['kind']),
                              color=COLORS[r['kind']], name=r['id'], smooth_shading=False)
        robots[r['id']] = actor
    task_actors = {}
    for task in data['tasks']:
        actor = plot.add_mesh(pv.Sphere(radius=.85, center=(0, 0, 0)),
                              color='#eabf3a', name='task_' + task['id'])
        actor.SetPosition(*task['position'])
        task_actors[task['id']] = actor

    # Right-side task list with progress, ownership and status at current replay time.
    sidebar = QFrame()
    sidebar.setMinimumWidth(265)
    sidebar.setMaximumWidth(340)
    sidebar.setFrameShape(QFrame.StyledPanel)
    side_layout = QVBoxLayout(sidebar)
    side_title = QLabel('Tasks · current frame')
    side_title.setStyleSheet('font-size: 15px; font-weight: bold;')
    side_layout.addWidget(side_title)
    task_summary = QLabel('')
    side_layout.addWidget(task_summary)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll_contents = QWidget()
    task_list = QVBoxLayout(scroll_contents)
    task_list.setAlignment(Qt.AlignTop)
    scroll.setWidget(scroll_contents)
    side_layout.addWidget(scroll, stretch=1)
    task_widgets = {}
    for tid, spec in task_specs.items():
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        name = QPushButton(f"{tid}  ·  {spec.get('type', 'task')}")
        name.setToolTip("Focus camera on this task and its assigned robot")
        name.setStyleSheet("text-align: left; font-weight: bold; padding: 3px;")
        status = QLabel('Unreleased')
        status.setWordWrap(True)
        progress = QProgressBar()
        progress.setRange(0, 1000)
        progress.setFormat('%p%')
        card_layout.addWidget(name)
        card_layout.addWidget(status)
        card_layout.addWidget(progress)
        task_list.addWidget(card)
        task_widgets[tid] = (card, name, status, progress)
    middle.addWidget(sidebar, stretch=1)

    controls = QHBoxLayout()
    vertical.addLayout(controls)
    clock = QLabel()
    controls.addWidget(clock)
    controls.addStretch()
    state = {'index':0, 'playing':False, 'camera':'3D', 'focused_task':None}
    completion_times = {e['task']: float(e['time']) for e in data.get('events', [])
                        if e.get('event') == 'task_complete' and 'task' in e}
    # For logs without completion events, infer order from recorded transitions.
    for frame in frames:
        for task in frame['tasks']:
            if task.get('status') == 'completed':
                completion_times.setdefault(task['id'], float(frame['time']))
    timer = QTimer(window)
    timer.setInterval(50)

    def set_view(mode):
        state['camera'] = mode
        state['focused_task'] = None
        center = np.array(bounds, dtype=float)/2
        distance = max(bounds)
        if mode == 'Top':
            plot.camera_position = [(center[0],center[1],distance*2), tuple(center), (0,1,0)]
        elif mode == 'Side':
            plot.camera_position = [(center[0],-distance*1.5,center[2]),tuple(center),(0,0,1)]
        elif mode == '3D':
            plot.camera_position = [(center[0]+distance*1.2,center[1]-distance*.8,distance*.85),
                                    tuple(center),(0,0,1)]
        # Free preserves the current camera and enables mouse orbit/pan/zoom.
        elif mode == 'Follow':
            focus_follow()
        plot.render()

    def focus_follow():
        if state['camera'] != 'Follow':
            return
        alive = [r for r in frames[state['index']]['robots'] if r['alive']]
        if alive:
            center = np.array(alive[0]['position'])
            plot.camera_position = [tuple(center+[-18,-22,16]),tuple(center),(0,0,1)]

    def reset_view():
        state['focused_task'] = None
        set_view('3D')

    def focus_task(tid):
        state['focused_task'] = tid
        state['camera'] = 'Task'
        focus_selected_task()
        plot.render()

    def focus_selected_task():
        tid = state['focused_task']
        if not tid:
            return
        frame = frames[state['index']]
        current = next((t for t in frame['tasks'] if t['id'] == tid), None)
        if not current or current.get('status') in ('unreleased', 'completed', 'expired'):
            reset_view()
            return
        task_xyz = np.asarray(task_specs[tid]['position'], dtype=float)
        owner = current.get('owner')
        agent = next((r for r in frame['robots'] if r['id'] == owner and r['alive']), None)
        points = [task_xyz]
        if agent is not None:
            points.append(np.asarray(agent['position'], dtype=float))
        low = np.min(points, axis=0)
        high = np.max(points, axis=0)
        center = (low + high) / 2
        # Fit both the task and executor into a 3D view, with a minimum zoom
        # span for single-point/unassigned tasks.
        span = max(float(np.max(high-low)), 22.0)
        distance = span * 1.8
        plot.camera_position = [tuple(center + np.array([distance, -distance, distance*.8])),
                                tuple(center), (0, 0, 1)]

    for tid, (card, name, status, progress) in task_widgets.items():
        name.clicked.connect(lambda checked=False, task_id=tid: focus_task(task_id))

    for name in ('Top','3D','Side','Follow','Free'):
        b = QPushButton(name)
        controls.addWidget(b)
        b.clicked.connect(lambda checked=False, mode=name: set_view(mode))
    reset_button = QPushButton('Reset View')
    reset_button.setToolTip('Restore the original overview camera')
    reset_row = QHBoxLayout()
    reset_row.addStretch()
    reset_row.addWidget(reset_button)
    reset_row.addStretch()
    viewport_layout.addLayout(reset_row)
    reset_button.clicked.connect(reset_view)
    play = QPushButton('Play / Pause')
    controls.addWidget(play)
    play.clicked.connect(lambda: state.update(playing=not state['playing']))
    seek = QSlider(Qt.Horizontal)
    seek.setRange(0, len(frames)-1)
    vertical.addWidget(seek)

    def update(idx):
        idx = max(0, min(int(idx), len(frames)-1))
        state['index'] = idx
        frame = frames[idx]
        for kind, (label, caption) in platform_legend.items():
            fleet = [r for r in frame['robots'] if r['kind'] == kind]
            # Busy means alive with a current assignment, including travelling
            # to the assigned task; destroyed and unassigned units are not busy.
            busy = sum(bool(r['alive']) and bool(r.get('task_id')) for r in fleet)
            label.setText(f'{caption}  {busy}/{len(fleet)}')
        for r in frame['robots']:
            actor = robots[r['id']]
            actor.SetPosition(*r['position'])
            actor.SetOrientation(0, 0, math.degrees(float(r.get('heading', 0))))
            actor.SetVisibility(bool(r['alive']))
        current_tasks = {t['id']:t for t in frame['tasks']}
        counts = {'completed':0, 'servicing':0, 'available':0}
        for tid, spec in task_specs.items():
            t = current_tasks.get(tid, {})
            task_status = t.get('status', 'unreleased')
            if task_status in counts:
                counts[task_status] += 1
            total = float(spec.get('total_work', 1))
            amount = float(t.get('progress', 0))
            remaining = task_remaining_scale(amount, total)
            # Actor scale multiplies the original sphere radius; scale=0 at completion.
            # Explicitly hide zero-radius markers to avoid degenerate render artifacts.
            task_actors[tid].SetScale(remaining, remaining, remaining)
            task_actors[tid].SetVisibility(
                task_status not in ('unreleased', 'expired', 'completed') and remaining > 0
            )
            card, name_button, status_label, progress = task_widgets[tid]
            name_button.setEnabled(task_status not in ('unreleased', 'completed', 'expired'))
            frac = max(0.0, min(1.0, amount/total)) if total > 0 else 0.0
            progress.setValue(round(frac*1000))
            owner = t.get('owner') or 'Unassigned'
            status_label.setText(f"{task_status.replace('_',' ').title()}  |  {owner}  |  {amount:.1f}/{total:g} units")
            status_label.setStyleSheet('color: #2a9d5b;' if task_status=='completed' else '')
        # Active/pending tasks first; completed tasks at bottom, latest completion first.
        def sort_key(tid):
            t = current_tasks.get(tid, {})
            s = t.get('status', 'unreleased')
            if s == 'completed':
                return (3, -completion_times.get(tid, 0.0), tid)
            return (0 if s in ('servicing', 'assigned', 'available') else
                    1 if s == 'unreleased' else 2, 0, tid)
        for pos, tid in enumerate(sorted(task_specs, key=sort_key)):
            task_list.insertWidget(pos, task_widgets[tid][0])
        task_summary.setText(f"{counts['completed']} completed · {counts['servicing']} servicing · {counts['available']} available")
        if state['camera'] == 'Task':
            focus_selected_task()
        else:
            focus_follow()
        clock.setText(f"t = {frame['time']:.1f} s  |  frame {idx+1}/{len(frames)}")
        seek.blockSignals(True)
        seek.setValue(idx)
        seek.blockSignals(False)
        plot.render()

    seek.valueChanged.connect(update)
    def step():
        if state['playing']:
            update(state['index']+1)
            if state['index'] >= len(frames)-1:
                state['playing'] = False
    timer.timeout.connect(step)
    timer.start()
    window.setCentralWidget(root)
    window.resize(1480,850)
    set_view('3D')
    update(0)
    window.show()
    return app.exec()

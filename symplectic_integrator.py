#!/usr/bin/env python3
"""
SYMPLECTIC INTEGRATOR FOR MUSICAL PHASE SPACE
==============================================

Theory: Music is a Hamiltonian system where:
  - q = harmonic position (pitch class on circle of fifths, 0-11)
  - p = rhythmic momentum (rate of harmonic change)
  - H(q,p) = T(p) + V(q) = kinetic + potential tension

Symplectic integrators conserve the symplectic 2-form dq ∧ dp BY CONSTRUCTION,
preserving phase space volume (Liouville's theorem) and bounding energy drift.
Non-symplectic integrators (Euler, RK4) show secular energy drift.
"""

import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# =============================================================================
# MUSICAL PHASE SPACE
# =============================================================================

class MusicalPhaseSpace:
    """2D phase space: q = harmonic position, p = rhythmic momentum."""
    
    def __init__(self, n_pitch_classes=12):
        self.n = n_pitch_classes
    
    def tension_potential(self, q):
        """V(q) = -cos(2πq/12). Min at fifths, max at tritone."""
        return -np.cos(2 * np.pi * q / self.n)
    
    def dV_dq(self, q):
        """dV/dq = (2π/12)*sin(2πq/12)."""
        return np.sin(2 * np.pi * q / self.n) * (2 * np.pi / self.n)
    
    def kinetic_energy(self, p):
        return 0.5 * p ** 2
    
    def hamiltonian(self, q, p):
        """H(q,p) = T(p) + V(q) = total tension."""
        return self.kinetic_energy(p) + self.tension_potential(q % self.n)
    
    def energy_surface(self, qr=(0,12), pr=(-3,3), n=100):
        Q = np.linspace(qr[0], qr[1], n)
        P = np.linspace(pr[0], pr[1], n)
        return np.meshgrid(Q, P) + (self.hamiltonian(*np.meshgrid(Q, P)),)

# =============================================================================
# INTEGRATOR STEP FUNCTIONS
# =============================================================================

def symplectic_euler_step(ps, q, p, dt):
    """1st-order symplectic: p←p-dt·dV/dq, q←q+dt·p_new. Conserves H̃ = H+O(dt)."""
    p_new = p - dt * ps.dV_dq(q)
    q_new = q + dt * p_new
    return q_new, p_new

def stormer_verlet_step(ps, q, p, dt):
    """2nd-order symplectic (leapfrog). Conserves H̃ = H+O(dt²)."""
    p_half = p - 0.5 * dt * ps.dV_dq(q)
    q_new = q + dt * p_half
    p_new = p_half - 0.5 * dt * ps.dV_dq(q_new)
    return q_new, p_new

def explicit_euler_step(ps, q, p, dt):
    """1st-order NOT symplectic: q←q+dt·p, p←p-dt·dV/dq. Secular drift."""
    q_new = q + dt * p
    p_new = p - dt * ps.dV_dq(q)
    return q_new, p_new

def rk4_step(ps, q, p, dt):
    """4th-order NOT symplectic. Secular energy drift."""
    def f(y):
        return np.array([y[1], -ps.dV_dq(y[0] % ps.n)])
    y = np.array([q, p])
    k1 = f(y)
    k2 = f(y + 0.5*dt*k1)
    k3 = f(y + 0.5*dt*k2)
    k4 = f(y + dt*k3)
    yn = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return yn[0], yn[1]

# =============================================================================
# DISPATCH
# =============================================================================

STEP_FUNCTIONS = {
    'symplectic_euler': symplectic_euler_step,
    'stormer_verlet': stormer_verlet_step,
    'explicit_euler': explicit_euler_step,
    'rk4': rk4_step,
}

ALL_METHODS = ['symplectic_euler', 'stormer_verlet', 'explicit_euler', 'rk4']

METHOD_LABELS = {
    'symplectic_euler': 'Symplectic Euler (1st)',
    'stormer_verlet': 'Störmer-Verlet (2nd)',
    'explicit_euler': 'Euler (non-symplectic)',
    'rk4': 'RK4 (non-symplectic)',
}

METHOD_COLORS = {
    'symplectic_euler': '#2196F3',
    'stormer_verlet': '#4CAF50',
    'explicit_euler': '#F44336',
    'rk4': '#FF9800',
}

METHOD_LS = {
    'symplectic_euler': '-',
    'stormer_verlet': '-',
    'explicit_euler': '--',
    'rk4': ':',
}

def integrate_trajectory(ps, q0, p0, dt, n_steps, method='symplectic_euler'):
    """Integrate a trajectory. Returns (q_traj, p_traj, H_traj)."""
    step_fn = STEP_FUNCTIONS[method]
    q = np.empty(n_steps + 1); p = np.empty(n_steps + 1); H = np.empty(n_steps + 1)
    q[0], p[0] = q0, p0
    H[0] = ps.hamiltonian(q0, p0)
    for i in range(n_steps):
        q[i+1], p[i+1] = step_fn(ps, q[i], p[i], dt)
        q[i+1] %= ps.n
        H[i+1] = ps.hamiltonian(q[i+1], p[i+1])
    return q, p, H

def phase_space_volume(q, p):
    """Area of convex hull in phase space."""
    pts = np.column_stack([q, p])
    if len(pts) < 3: return 0.0
    try: return ConvexHull(pts).volume
    except Exception: return 0.0

def liouville_violation(q, p, ws=50):
    """Normalized variance of phase-space volume across windows (0 = perfect)."""
    n = len(q)
    if n < 2*ws: return float('nan')
    vols = []
    for i in range(0, n-ws, ws//2):
        v = phase_space_volume(q[i:i+ws], p[i:i+ws])
        if v > 1e-15: vols.append(v)
    if len(vols) < 3: return float('nan')
    return float(np.var(vols) / max(np.mean(vols), 1e-30))

def energy_drift_metrics(H):
    """Energy drift statistics."""
    dH = H - H[0]
    return {
        'mean_drift': float(np.mean(dH)),
        'max_drift': float(np.max(np.abs(dH))),
        'final_drift': float(dH[-1]),
        'drift_rate': float(np.polyfit(np.arange(len(dH)), dH, 1)[0]),
        'rms_error': float(np.sqrt(np.mean(dH**2))),
    }

# =============================================================================
# CHORD PROGRESSIONS
# =============================================================================

PC_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

DIATONIC = { # fifths-order pos → (function, quality, intervals)
    0:  ('I',  'maj',  [0,4,7]),
    2:  ('ii', 'min',  [0,3,7]),
    4:  ('iii','min',  [0,3,7]),
    5:  ('IV', 'maj',  [0,4,7]),
    7:  ('V',  'dom7', [0,4,7,10]),
    9:  ('vi', 'min',  [0,3,7]),
    11: ('vii°','dim', [0,3,6]),
}

def pos_to_chord(q):
    """Convert harmonic position q to (pcset, root, quality, name)."""
    qm = q % 12
    pc = (5 * int(round(qm))) % 12
    best = ('?','maj',[0,4,7])
    best_d = 1.0
    for pos, (func, qual, ivs) in DIATONIC.items():
        d = min(abs(qm - pos), 12 - abs(qm - pos))
        if d < best_d: best_d = d; best = (func, qual, ivs)
    func, quality, intervals = best
    pcset = sorted([(r + 0) % 12 for r in intervals])  # key center = 0
    root = pc % 12
    name = f"{PC_NAMES[root]}{quality}"
    return pcset, root, quality, name

def traj_to_progression(q_traj):
    """Sample q_traj at chord changes."""
    prog = []
    prev = None
    for q in q_traj:
        _, r, qn, n = pos_to_chord(q)
        if prev is None or r != prev:
            prog.append((_, r, qn, n))
            prev = r
    return prog

def prog_str(p): return ' → '.join(n for _,_,_,n in p)

def compute_coherence(prog, key=0):
    """Score 0-1 for harmonic coherence."""
    if len(prog) < 2: return {'overall_score': 1.0}
    dists = []; diat = 0; drastic = 0
    diaroots = {(5*d) % 12 for d in DIATONIC}
    for i,(_,r,_,_) in enumerate(prog):
        dists.append(min(abs(r - key), 12 - abs(r - key)))
        if r in diaroots: diat += 1
        if i > 0:
            c = min(abs(r - prog[i-1][1]), 12 - abs(r - prog[i-1][1]))
            if c > 3: drastic += 1
    md = np.mean(dists)
    c = {'mean_distance_from_key': float(md),
         'diatonic_ratio': float(diat/len(prog)),
         'drastic_changes_ratio': float(drastic/max(len(prog)-1,1))}
    c['overall_score'] = float(c['diatonic_ratio']*0.4 +
                                (1-c['drastic_changes_ratio'])*0.3 +
                                max(0,1-md/6)*0.3)
    return c

def progression_to_midi(prog, fn, tempo=100):
    """Save progression as MIDI."""
    import mido
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    mid = MidiFile()
    tr = MidiTrack(); mid.tracks.append(tr)
    tr.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
    tr.append(MetaMessage('time_signature', numerator=4, denominator=4))
    tpc = 480 * 4  # ticks per chord
    for pcset,_,_,_ in prog:
        notes = [max(36, min(96, 60 + pc + 12)) for pc in pcset]
        for n in notes: tr.append(Message('note_on', note=n, velocity=70, time=0))
        for ni, n in enumerate(notes):
            tr.append(Message('note_off', note=n, velocity=64, time=tpc if ni==0 else 0))
    mid.save(fn); return fn

# =============================================================================
# EXPERIMENT 1: ENERGY CONSERVATION
# =============================================================================

def run_experiment_energy(ps, odir):
    print("="*70)
    print("EXPERIMENT 1: ENERGY CONSERVATION")
    print("="*70)
    conds = [
        (0.0, 0.5, 0.05, 10000, "Slow (q₀=0, p₀=0.5)"),
        (3.0, 1.5, 0.05, 10000, "Medium (q₀=3, p₀=1.5)"),
        (1.0, 3.0, 0.05, 10000, "Fast (q₀=1, p₀=3.0)"),
        (6.0, 0.1, 0.05, 10000, "Near tritone (q₀=6)"),
        (7.0, 2.0, 0.01, 50000, "High resolution (dt=0.01)"),
    ]
    R = {}
    for qi,(q0,p0,dt,ns,label) in enumerate(conds):
        ck = f"cond_{qi}"
        print(f"\n  Condition {qi}: {label}")
        R[ck] = {'label': label, 'q0': q0, 'p0': p0, 'dt': dt, 'n_steps': ns}
        for m in ALL_METHODS:
            _,_,H = integrate_trajectory(ps, q0, p0, dt, ns, m)
            d = energy_drift_metrics(H)
            R[ck][m] = d
            print(f"    {METHOD_LABELS[m]:30s}  final={d['final_drift']:+8.1e}  max|d|={d['max_drift']:8.1e}  rate={d['drift_rate']:+8.1e}")
    return R

def plot_energy_drift(ps, R, odir):
    ckeys = sorted([k for k in R if k.startswith('cond_')])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ci, ck in enumerate(ckeys[:4]):
        ax = axes[ci]; c = R[ck]
        for m in ALL_METHODS:
            _,_,H = integrate_trajectory(ps, c['q0'], c['p0'], c['dt'], c['n_steps'], m)
            dH = H - H[0]; t = np.arange(len(H))*c['dt']
            s = max(1, len(H)//2000)
            ax.plot(t[::s], dH[::s], color=METHOD_COLORS[m], ls=METHOD_LS[m], lw=0.7, label=METHOD_LABELS[m], alpha=0.8)
        ax.set_xlabel('Time'); ax.set_ylabel('ΔH')
        ax.set_title(f'Energy Drift — {c["label"]}')
        ax.legend(fontsize=7, loc='lower right'); ax.grid(True, alpha=0.3)
    plt.suptitle('Energy Conservation: Symplectic vs Non-Symplectic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'energy_drift.png'), dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: energy_drift.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    clabels = [R[k]['label'].split('(')[0].strip() for k in ckeys]
    x = np.arange(len(ckeys)); w = 0.2
    for mi, m in enumerate(ALL_METHODS):
        vals = [max(abs(R[k][m]['final_drift']), 1e-30) for k in ckeys]
        ax.bar(x + mi*w, vals, w, color=METHOD_COLORS[m], label=METHOD_LABELS[m], alpha=0.85)
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(clabels, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('|Final Energy Drift|'); ax.set_title('Final Energy Drift Across Conditions')
    ax.set_yscale('log'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'energy_drift_bar.png'), dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: energy_drift_bar.png")

    ck = ckeys[1]; c = R[ck]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)); axes = axes.flatten()
    for mi, m in enumerate(ALL_METHODS):
        ax = axes[mi]
        q,p,_ = integrate_trajectory(ps, c['q0'], c['p0'], c['dt'], min(c['n_steps'], 500), m)
        ax.plot(q, p, color=METHOD_COLORS[m], lw=0.8, alpha=0.7)
        ax.scatter(q[0], p[0], c='green', s=60, marker='o', zorder=5, label='Start')
        ax.scatter(q[-1], p[-1], c='red', s=60, marker='x', zorder=5, label='End')
        ax.set_xlabel('q (harmonic position)'); ax.set_ylabel('p (rhythmic momentum)')
        ax.set_title(METHOD_LABELS[m]); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle('Phase Space Trajectories (First 500 Steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'phase_space_trajectories.png'), dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: phase_space_trajectories.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    Q, P, H = ps.energy_surface()
    cf = ax.contourf(Q, P, H, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(cf, ax=ax, label='H(q,p) = Total Tension')
    for m in ALL_METHODS:
        q,p,_ = integrate_trajectory(ps, c['q0'], c['p0'], c['dt'], 500, m)
        ax.plot(q, p, color=METHOD_COLORS[m], lw=0.6, alpha=0.7, label=METHOD_LABELS[m])
    ax.scatter(c['q0'], c['p0'], c='white', s=100, marker='*', edgecolors='black', zorder=10, label='Start')
    ax.set_xlabel('q'); ax.set_ylabel('p')
    ax.set_title('Energy Surface with Trajectory Overlay'); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'energy_surface.png'), dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: energy_surface.png")

# =============================================================================
# EXPERIMENT 2: LONG-TERM STABILITY (100k steps)
# =============================================================================

def run_experiment_longterm(ps, odir):
    print("\n"+"="*70)
    print("EXPERIMENT 2: LONG-TERM STABILITY (100,000 steps)")
    print("="*70)
    q0, p0, dt = 2.0, 1.0, 0.05
    ns = 100000
    R = {}
    print(f"\n  q₀={q0}, p₀={p0}, dt={dt}, steps={ns:,}")
    for m in ALL_METHODS:
        q,p,H = integrate_trajectory(ps, q0, p0, dt, ns, m)
        d = energy_drift_metrics(H)
        vol = phase_space_volume(q[-2000:], p[-2000:])
        R[m] = {'q': q.tolist(), 'p': p.tolist(), 'H': H.tolist(),
                'drift': d, 'final_volume': float(vol)}
        print(f"  {METHOD_LABELS[m]:30s}  final={d['final_drift']:+9.2e}  max|d|={d['max_drift']:9.2e}  rate={d['drift_rate']:+9.2e}  vol={vol:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for m in ALL_METHODS:
        H = np.array(R[m]['H']); dH = H - H[0]; t = np.arange(len(H))*dt
        s = len(H)//2000
        axes[0,0].plot(t[::s], dH[::s], color=METHOD_COLORS[m], lw=0.5, label=METHOD_LABELS[m], alpha=0.8)
    axes[0,0].set_xlabel('Time'); axes[0,0].set_ylabel('ΔH'); axes[0,0].set_title('Energy Drift — 100k steps')
    axes[0,0].legend(fontsize=7); axes[0,0].grid(True, alpha=0.3)

    for m in ALL_METHODS:
        q = np.array(R[m]['q'][-2000:]); p = np.array(R[m]['p'][-2000:])
        axes[0,1].plot(q, p, color=METHOD_COLORS[m], lw=0.3, alpha=0.5, label=METHOD_LABELS[m])
    axes[0,1].set_xlabel('q'); axes[0,1].set_ylabel('p')
    axes[0,1].set_title('Phase Space — Final 2000 Steps'); axes[0,1].legend(fontsize=7); axes[0,1].grid(True, alpha=0.3)

    x = np.arange(4); w = 0.25
    fd = [abs(R[m]['drift']['final_drift']) for m in ALL_METHODS]
    md = [R[m]['drift']['max_drift'] for m in ALL_METHODS]
    dr = [abs(R[m]['drift']['drift_rate']) for m in ALL_METHODS]
    axes[1,0].bar(x - w, fd, w, color=[METHOD_COLORS[m] for m in ALL_METHODS], alpha=0.85, label='|Final|')
    axes[1,0].bar(x, md, w, color=[METHOD_COLORS[m] for m in ALL_METHODS], alpha=0.5, label='Max| |')
    axes[1,0].bar(x + w, dr, w, color=[METHOD_COLORS[m] for m in ALL_METHODS], alpha=0.3, label='|Rate|')
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels([METHOD_LABELS[m] for m in ALL_METHODS], fontsize=8)
    axes[1,0].set_ylabel('Energy Error'); axes[1,0].set_title('Error Comparison (100k)')
    axes[1,0].set_yscale('log'); axes[1,0].legend(fontsize=7); axes[1,0].grid(True, alpha=0.3, axis='y')

    for m in ALL_METHODS:
        H = np.array(R[m]['H']); dH = H - H[0]; t = np.arange(len(H))*dt
        axes[1,1].plot(t, dH, color=METHOD_COLORS[m], lw=0.5, alpha=0.8, label=METHOD_LABELS[m])
    axes[1,1].set_xlabel('Time'); axes[1,1].set_ylabel('ΔH')
    axes[1,1].set_title('Full Drift (no subsample)'); axes[1,1].legend(fontsize=7); axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('Long-Term Stability: 100,000 Step Integration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'long_term_stability.png'), dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: long_term_stability.png")
    return R

# =============================================================================
# EXPERIMENT 3: CHORD PROGRESSIONS
# =============================================================================

def run_experiment_chords(ps, odir):
    print("\n"+"="*70)
    print("EXPERIMENT 3: SYMPLECTIC CHORD PROGRESSIONS")
    print("="*70)
    configs = [(0.0, 0.3), (2.0, 0.8), (5.0, 1.2), (7.0, 0.5)]
    dt, ns = 0.08, 1500

    all_coherence = {m: [] for m in ALL_METHODS}
    all_progs = {}

    for qi, (q0, p0) in enumerate(configs):
        key = f"Q{q0}_P{p0}"
        print(f"\n  Starting from q₀={q0}, p₀={p0}:")
        for m in ALL_METHODS:
            q,_,_ = integrate_trajectory(ps, q0, p0, dt, ns, m)
            prog = traj_to_progression(q)
            coh = compute_coherence(prog)
            all_coherence[m].append(coh['overall_score'])
            all_progs[f"{key}_{m}"] = {'progression': [(p_,r,q_,n) for p_,r,q_,n in prog],
                                        'coherence': coh, 'method': m}
            print(f"    {METHOD_LABELS[m]:30s}  chords={len(prog):3d}  coherence={coh['overall_score']:.3f}  {prog_str(prog[:8])}...")

    # Summary stats
    print("\n  Coherence Summary (mean ± std across all configs):")
    for m in ALL_METHODS:
        scores = all_coherence[m]
        print(f"    {METHOD_LABELS[m]:30s}  {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Comparison stats
    symp_scores = all_coherence['symplectic_euler'] + all_coherence['stormer_verlet']
    nonsymp_scores = all_coherence['explicit_euler'] + all_coherence['rk4']
    ratio = np.mean(symp_scores) / max(np.mean(nonsymp_scores), 1e-10)
    print(f"\n  Symplectic vs Non-Symplectic coherence ratio: {ratio:.3f}x")

    # Save best progression as MIDI
    best_key = max(all_progs, key=lambda k: all_progs[k]['coherence']['overall_score'])
    best_prog = all_progs[best_key]
    midi_fn = os.path.join(odir, f"best_progression_{best_key}.mid")
    progression_to_midi(best_prog['progression'], midi_fn)
    print(f"\n  Best progression ({best_key}):")
    print(f"    Coherence: {best_prog['coherence']['overall_score']:.3f}")
    print(f"    Progression: {prog_str(best_prog['progression'])}")
    print(f"    MIDI saved: {midi_fn}")

    # Plot coherence comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs)); w = 0.2
    for mi, m in enumerate(ALL_METHODS):
        scores = [all_progs[f"Q{q0}_P{p0}_{m}"]['coherence']['overall_score']
                  for (q0,p0) in configs]
        ax.bar(x + mi*w, scores, w, color=METHOD_COLORS[m], label=METHOD_LABELS[m], alpha=0.85)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([f"q₀={q0}, p₀={p0}" for q0,p0 in configs], fontsize=8)
    ax.set_ylabel('Coherence Score'); ax.set_title('Chord Progression Coherence by Integrator')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'chord_progression_coherence.png'), dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: chord_progression_coherence.png")

    # Bar chart showing mean coherence
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(all_coherence[m]) for m in ALL_METHODS]
    stds = [np.std(all_coherence[m]) for m in ALL_METHODS]
    colors = [METHOD_COLORS[m] for m in ALL_METHODS]
    ax.bar(range(4), means, 0.5, yerr=stds, color=colors, alpha=0.85, capsize=5)
    ax.set_xticks(range(4)); ax.set_xticklabels([METHOD_LABELS[m] for m in ALL_METHODS], fontsize=9)
    ax.set_ylabel('Mean Coherence Score'); ax.set_title('Mean Chord Progression Coherence')
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(odir, 'mean_coherence.png'), dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: mean_coherence.png")

    return all_progs, all_coherence

# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = "/home/phoenix/.openclaw/workspace/experiments/symplectic-music"
    os.makedirs(output_dir, exist_ok=True)

    ps = MusicalPhaseSpace()
    print(f"{'='*70}")
    print(f"MUSICAL PHASE SPACE SYMPLECTIC INTEGRATOR")
    print(f"{'='*70}")
    print(f"  Hamiltonian: H(q,p) = ½p² - cos(2πq/12)")
    print(f"  dq/dt = ∂H/∂p = p")
    print(f"  dp/dt = -∂H/∂q = -(2π/12)·sin(2πq/12)")
    print(f"  V(q) minimum at perfect fifths, maximum at tritone")
    print(f"  Symplectic integrators conserve dq ∧ dp BY CONSTRUCTION\n")

    # Experiment 1
    R1 = run_experiment_energy(ps, output_dir)
    plot_energy_drift(ps, R1, output_dir)

    # Experiment 2
    R2 = run_experiment_longterm(ps, output_dir)

    # Experiment 3
    R3, coh = run_experiment_chords(ps, output_dir)

    # Save all results
    results = {
        'energy_conservation': R1,
        'long_term': {m: {k: R2[m][k] for k in R2[m] if k != 'q' and k != 'p' and k != 'H'}
                       for m in ALL_METHODS},
        'coherence': {m: {'scores': coh[m], 'mean': float(np.mean(coh[m])),
                          'std': float(np.std(coh[m]))} for m in ALL_METHODS},
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir}/results.json")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n1. Energy Conservation (10,000 steps):")
    for ck in sorted([k for k in R1 if k.startswith('cond_')]):
        c = R1[ck]
        print(f"  {c['label']}:")
        for m in ALL_METHODS:
            d = c[m]
            print(f"    {METHOD_LABELS[m]:30s}  final={d['final_drift']:+8.1e}  max={d['max_drift']:8.1e}  rate={d['drift_rate']:+8.1e}")

    print("\n2. Long-term Stability (100,000 steps):")
    for m in ALL_METHODS:
        d = R2[m]['drift']
        print(f"    {METHOD_LABELS[m]:30s}  final={d['final_drift']:+9.2e}  max|d|={d['max_drift']:9.2e}  rate={d['drift_rate']:+9.2e}")

    print("\n3. Chord Progression Coherence:")
    for m in ALL_METHODS:
        print(f"    {METHOD_LABELS[m]:30s}  mean={np.mean(coh[m]):.3f} ± {np.std(coh[m]):.3f}")
    symp_mean = np.mean(coh['symplectic_euler'] + coh['stormer_verlet'])
    nonsymp_mean = np.mean(coh['explicit_euler'] + coh['rk4'])
    print(f"    Symplectic / Non-Symplectic ratio: {symp_mean/max(nonsymp_mean,1e-10):.3f}x")

    print(f"\nPlots saved in: {output_dir}")
    print("="*70)

if __name__ == '__main__':
    main()

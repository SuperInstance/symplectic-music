#!/usr/bin/env python3
"""Liouville volume-preservation analysis for symplectic music integrators."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symplectic_integrator import *
import matplotlib.pyplot as plt
import numpy as np

odir = os.path.dirname(os.path.abspath(__file__))
ps = MusicalPhaseSpace()

print("="*70)
print("LIOUVILLE'S THEOREM: PHASE SPACE VOLUME PRESERVATION")
print("="*70)

# Test: integrate with multiple trajectories from random initial conditions
# and measure how phase space volume evolves
q0s = np.linspace(0, 11, 6)
p0s = np.linspace(-2, 2, 5)
dt = 0.05
ns = 5000

# For each method, compute the volume evolution over time
window = 200

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for mi, method in enumerate(['symplectic_euler', 'stormer_verlet', 'explicit_euler']):
    ax = axes[mi]
    vol_evolution = []
    
    for q0 in q0s:
        for p0 in p0s:
            q, p, H = integrate_trajectory(ps, q0, p0, dt, ns, method)
            
            # Compute volume over sliding windows
            vols = []
            for i in range(0, ns - window, window // 2):
                v = phase_space_volume(q[i:i+window], p[i:i+window])
                if v > 1e-15:
                    vols.append(v)
            
            vols = np.array(vols)
            if len(vols) > 1:
                vols = vols / vols[0]  # Normalize to initial volume
                ax.plot(np.arange(len(vols)) * (window//2) * dt, vols, 
                        color=METHOD_COLORS[method], alpha=0.3, lw=0.5)
                vol_evolution.append(vols)
    
    if vol_evolution:
        mean_vol = np.mean(vol_evolution, axis=0)
        std_vol = np.std(vol_evolution, axis=0)
        t = np.arange(len(mean_vol)) * (window//2) * dt
        ax.plot(t, mean_vol, color=METHOD_COLORS[method], lw=2, label='Mean')
        ax.fill_between(t, mean_vol - std_vol, mean_vol + std_vol, 
                        color=METHOD_COLORS[method], alpha=0.1)
    
    ax.axhline(y=1.0, color='gray', ls='--', lw=1, alpha=0.5, label='Expected (Liouville)')
    ax.set_xlabel('Time'); ax.set_ylabel('V(t) / V(0)')
    ax.set_title(f'{METHOD_LABELS[method]}')
    ax.set_ylim(0, 5)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('Liouville\'s Theorem: Phase Space Volume Conservation\n(Symplectic methods should maintain V(t)/V(0) ≈ 1)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(odir, 'liouville_volume.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: liouville_volume.png")

# =============================================================================
# ALSO: Phase space area change per step (determinant test)
# =============================================================================
print("\n  Jacobian determinant test (mean |det(J) - 1| per step):")
for method in ALL_METHODS:
    q, p, _ = integrate_trajectory(ps, 1.0, 1.0, 0.05, 100, method)
    dets = []
    step_fn = STEP_FUNCTIONS[method]
    for i in range(99):
        # Numerical Jacobian: perturb q,p by epsilon, measure area change
        eps = 1e-6
        q1, p1 = step_fn(ps, q[i] + eps, p[i], 0.05)
        q2, p2 = step_fn(ps, q[i], p[i] + eps, 0.05)
        # Area change = det(J) = (∂q'/∂q)(∂p'/∂p) - (∂q'/∂p)(∂p'/∂q)
        dq_dq = (q1 - step_fn(ps, q[i], p[i], 0.05)[0]) / eps
        dp_dq = (p1 - step_fn(ps, q[i], p[i], 0.05)[1]) / eps
        dq_dp = (q2 - step_fn(ps, q[i], p[i], 0.05)[0]) / eps
        dp_dp = (p2 - step_fn(ps, q[i], p[i], 0.05)[1]) / eps
        det = dq_dq * dp_dp - dq_dp * dp_dq
        dets.append(abs(det - 1.0))
    print(f"    {METHOD_LABELS[method]:30s}  mean |det(J)-1| = {np.mean(dets):.4e}  (0 = perfect Liouville)")

print("\n  Done.")

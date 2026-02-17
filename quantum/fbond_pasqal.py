#!/usr/bin/env python3
"""
F_bond on Pasqal — Entanglement Measurement for Molecular Clusters
===================================================================

Maps molecular orbital entanglement onto a Rydberg atom register
and measures entanglement entropy S_E directly from quantum hardware
to compute F_bond(A) without classical diagonalization.

Device: AnalogDevice (Fresnel QPU compatible)
  - max_amp: 2π×2 rad/μs, max_det: 2π×20 rad/μs
  - min atom distance: 5 μm, max radial: 38 μm
  - max sequence duration: 6000 ns

Reuses authentication infrastructure from the Wormhole/Pasqal project.

Requirements
------------
    pip install pulser pasqal-cloud numpy

Credentials
-----------
    export PASQAL_PROJECT_ID="your-project-id"
    export PASQAL_USERNAME="your-email"
    export PASQAL_PASSWORD="your-password"

Usage
-----
    python fbond_pasqal.py [--emulator EMU_FREE|EMU_MPS] [--shots 1000]
"""

import os
import sys
import time
import json
import argparse
from scipy.spatial.distance import pdist
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Pasqal / Pulser imports
# ---------------------------------------------------------------------------
try:
    from pulser import Pulse, Sequence, Register
    from pulser.devices import AnalogDevice
    from pulser.waveforms import RampWaveform, ConstantWaveform
except ImportError:
    print("ERROR: pulser not installed.  pip install pulser")
    sys.exit(1)

try:
    from pasqal_cloud import SDK
except ImportError:
    SDK = None  # Allow local-only mode without cloud submission


# ============================================================================
# 1. MOLECULAR SYSTEMS — coordinates from CCSD/def2-SVP calculations
# ============================================================================

# AnalogDevice hardware constraints (from pulser v1.7.0)
DEVICE_MIN_DISTANCE_UM = 5.0    # minimum inter-atom distance
DEVICE_MAX_RADIAL_UM = 38.0     # max distance from center
DEVICE_MAX_AMP = 2 * np.pi * 2  # 12.566 rad/μs (rydberg_global)
DEVICE_MAX_DET = 2 * np.pi * 20 # 125.66 rad/μs
DEVICE_MAX_DURATION = 6000      # ns

SYSTEMS = {
    "Al4_aromatic": {
        "description": "Al4(2-), D4h square planar, aromatic (2π e⁻)",
        "charge": -2,
        "coords_angstrom": np.array([
            [-1.270, -1.270, 0.0],
            [ 1.270, -1.270, 0.0],
            [ 1.270,  1.270, 0.0],
            [-1.270,  1.270, 0.0],
        ]),
        # Classical reference values (CCSD/def2-SVP)
        "classical_Fbond_B": 3.84,
        "classical_fe": 0.071,
        "classical_SE_max": 0.028,
    },

    "Al4_antiaromatic": {
        "description": "Al4(4-), D2h rectangular, antiaromatic (4π e⁻)",
        "charge": -4,
        "coords_angstrom": np.array([
            [-1.400, -1.100, 0.0],
            [ 1.400, -1.100, 0.0],
            [ 1.400,  1.100, 0.0],
            [-1.400,  1.100, 0.0],
        ]),
        "classical_Fbond_B": 4.03,
        "classical_fe": 0.072,
        "classical_SE_max": 0.019,
    },

    "B12_planar": {
        "description": "B12, D3h planar sheet",
        "charge": 0,
        "coords_angstrom": np.array([
            # Approximate D3h planar B12 ring
            [1.70 * np.cos(i * np.pi / 6), 1.70 * np.sin(i * np.pi / 6), 0.0]
            for i in range(12)
        ]),
        "classical_Fbond_B": 4.42,
        "classical_fe": 0.074,
        "classical_SE_max": 0.030,
    },

    "B6N6_planar": {
        "description": "B6N6, alternating borazine-like ring",
        "charge": 0,
        "coords_angstrom": np.array([
            # Alternating B-N hexagonal ring
            [1.44 * np.cos(i * np.pi / 6), 1.44 * np.sin(i * np.pi / 6), 0.0]
            for i in range(12)
        ]),
        "classical_Fbond_B": 5.11,
        "classical_fe": 0.071,
        "classical_SE_max": 0.035,
    },

    "Cs3Al8": {
        "description": "Cs3Al8(-), superatom cluster",
        "charge": -1,
        "coords_angstrom": np.array([
            # Al core (8 atoms) — from your B3LYP/def2-SVP geometry
            [ 0.000,  0.000,  0.000],
            [ 2.534,  0.000,  0.000],
            [ 1.267,  2.195,  0.000],
            [-1.267,  2.195,  0.000],
            [-2.534,  0.000,  0.000],
            [-1.267, -2.195,  0.000],
            [ 1.267, -2.195,  0.000],
            [ 0.000,  0.000,  2.534],
            # Cs shell (3 atoms)
            [ 3.800,  2.200,  3.000],
            [-3.800,  2.200,  3.000],
            [ 0.000, -4.400,  3.000],
        ]),
        "classical_Fbond_B": 5.58,
        "classical_fe": 0.042,
        "classical_SE_max": 0.013,
    },
}


# ============================================================================
# 2. PULSE SEQUENCE BUILDER
# ============================================================================

def project_3d_to_2d(coords_3d):
    """
    Project 3D coordinates to 2D using PCA, preserving inter-atom distances
    as much as possible. Applies random rotation first to avoid perfect alignment
    collapsing (e.g. axial atoms eclipsing central ones).
    """
    if coords_3d.shape[1] == 2:
        return coords_3d

    # Check if z-range is negligible (flat molecule)
    z_range = np.ptp(coords_3d[:, 2])
    if z_range < 1e-6:
        return coords_3d[:, :2]

    # Random rotation to break symmetry (avoiding perfect overlap)
    # Fixed seed for reproducibility
    rng = np.random.default_rng(42)
    
    # 3 random Euler angles
    alpha, beta, gamma = rng.uniform(0, 2*np.pi, 3)
    
    # Rotation matrices
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha),  np.cos(alpha), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(gamma), -np.sin(gamma)],
                   [0, np.sin(gamma),  np.cos(gamma)]])
    
    # Apply rotation
    rotated = coords_3d @ (Rz @ Ry @ Rx).T

    # PCA: project onto the 2 directions of maximum variance
    centered = rotated - rotated.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Take the two eigenvectors with largest eigenvalues
    idx = np.argsort(eigvals)[::-1][:2]
    basis = eigvecs[:, idx]
    projected = centered @ basis

    # Post-projection nudge: Force-directed layout (Spring Embedder)
    # Goal: Ensure all atoms are reasonably separated (e.g. > 1.0 A) in 2D
    # without destroying the overall topology.
    target_min_dist = 1.5  # Angstroms (in 2D projection)
    
    for _ in range(200):
        dists = pdist(projected)
        min_d = dists.min()
        if min_d > target_min_dist * 0.95:
            break
        
        # repulsive force between close atoms
        N = len(projected)
        forces = np.zeros_like(projected)
        
        for i in range(N):
            for j in range(i+1, N):
                delta = projected[i] - projected[j]
                r = np.linalg.norm(delta)
                
                # If too close, push apart strongly
                if r < target_min_dist:
                    # Force proportional to overlap
                    # Add small epsilon to avoid divide-by-zero
                    f_mag = (target_min_dist - r) * 0.5
                    if r < 1e-3:
                        # Identical points: push in random direction
                        direction = rng.random(2) - 0.5
                        direction /= np.linalg.norm(direction)
                    else:
                        direction = delta / r
                    
                    f = direction * f_mag
                    forces[i] += f
                    forces[j] -= f
                    
        # Apply forces with damping
        projected += forces * 0.5

    return projected


def auto_scale(coords_2d, min_dist_um=DEVICE_MIN_DISTANCE_UM,
               max_radial_um=DEVICE_MAX_RADIAL_UM, safety=1.05):
    """
    Compute scale factor so all atom pairs are ≥ min_dist_um apart
    and all atoms fit within max_radial_um from center.
    """
    from scipy.spatial.distance import pdist
    dists = pdist(coords_2d)
    min_pair_dist = dists.min() if len(dists) > 0 else 1.0

    # Scale for minimum distance
    scale_min = (min_dist_um * safety) / min_pair_dist

    # Scale for max radial distance
    centered = coords_2d - coords_2d.mean(axis=0)
    max_r = np.max(np.linalg.norm(centered, axis=1))
    scale_max = max_radial_um / (max_r if max_r > 0 else 1.0)
    
    # If scale_min violates max_radial, we must shrink (clip)
    # This means some atoms will be < min_dist_um apart (but hopefully close)
    if scale_min * max_r > max_radial_um:
        print(f"    ⚠ Cannot satisfy min_dist ({min_dist_um}μm) within max_radial ({max_radial_um}μm)")
        print(f"      Clipping scale to max possible ({scale_max:.2f}). Some pairs may be too close.")
        scale = scale_max
    else:
        scale = scale_min

    return scale


def build_fbond_sequence(coords_angstrom, ramp_time_ns=2000, hold_time_ns=500):
    """
    Build a Pasqal Pulser sequence for F_bond entanglement measurement.
    
    Handles 3D→2D projection and auto-scaling to satisfy device constraints.

    Parameters
    ----------
    coords_angstrom : array (N, 2) or (N, 3)
        Atom positions in Ångströms.
    ramp_time_ns : int
        Adiabatic ramp duration in nanoseconds.
    hold_time_ns : int
        Hold time at target Hamiltonian in nanoseconds.

    Returns
    -------
    Sequence
        Pulser sequence ready for serialization.
    float
        Scale factor used (μm/Å).
    """
    # Ensure total duration fits device limit
    total_dur = ramp_time_ns + hold_time_ns
    if total_dur > DEVICE_MAX_DURATION:
        ramp_time_ns = int(DEVICE_MAX_DURATION * 0.8)
        hold_time_ns = DEVICE_MAX_DURATION - ramp_time_ns
        print(f"    Clamped durations: ramp={ramp_time_ns}, hold={hold_time_ns} ns")

    # 3D → 2D projection
    coords_2d = project_3d_to_2d(coords_angstrom)

    # Auto-scale to satisfy device constraints
    scale = auto_scale(coords_2d)
    coords_um = coords_2d * scale

    # Center the register
    coords_um -= coords_um.mean(axis=0)

    print(f"    Scale: {scale:.2f} μm/Å, min pair dist: "
          f"{np.min(pdist(coords_um)):.1f} μm, max radial: "
          f"{np.max(np.linalg.norm(coords_um, axis=1)):.1f} μm")

    # Build register
    qubits = {f"q{i}": tuple(coords_um[i]) for i in range(len(coords_um))}
    reg = Register(qubits)

    # Build sequence on AnalogDevice
    seq = Sequence(reg, AnalogDevice)
    seq.declare_channel("rydberg", "rydberg_global")

    # Use device-safe amplitude and detuning
    omega = DEVICE_MAX_AMP * 0.95   # 95% of max to stay safe
    delta_start = -DEVICE_MAX_DET * 0.8  # 80% of max detuning
    delta_end = 0.0

    # Ensure durations are multiples of clock_period (4 ns)
    ramp_time_ns = (ramp_time_ns // 4) * 4
    hold_time_ns = (hold_time_ns // 4) * 4
    ramp_time_ns = max(ramp_time_ns, 16)  # min_duration = 16
    hold_time_ns = max(hold_time_ns, 16)

    # Ramp pulse: constant amplitude, sweeping detuning
    ramp_pulse = Pulse(
        amplitude=ConstantWaveform(ramp_time_ns, omega),
        detuning=RampWaveform(ramp_time_ns, delta_start, delta_end),
        phase=0.0,
    )
    seq.add(ramp_pulse, "rydberg")

    # Hold pulse: both constant at target
    hold_pulse = Pulse(
        amplitude=ConstantWaveform(hold_time_ns, omega),
        detuning=ConstantWaveform(hold_time_ns, delta_end),
        phase=0.0,
    )
    seq.add(hold_pulse, "rydberg")

    # Measure in ground-rydberg basis
    seq.measure("ground-rydberg")

    return seq, scale


# ============================================================================
# 3. ENTANGLEMENT EXTRACTION
# ============================================================================

def extract_entanglement(bitstrings, n_atoms):
    """
    Compute entanglement entropy and mutual information from bitstrings.

    Parameters
    ----------
    bitstrings : list of str
        Measurement outcomes, e.g. ["0010", "1001", ...].
    n_atoms : int
        Number of atoms/qubits.

    Returns
    -------
    dict
        Entropies, mutual information, S_E_max, etc.
    """
    n_shots = len(bitstrings)
    eps = 1e-15

    # --- Single-site entropies ---
    entropies = []
    for site in range(n_atoms):
        p1 = sum(1 for bs in bitstrings if bs[site] == '1') / n_shots
        p0 = 1.0 - p1
        p0 = max(eps, min(1 - eps, p0))
        p1 = max(eps, min(1 - eps, p1))
        S_i = -(p0 * np.log(p0) + p1 * np.log(p1))
        entropies.append(float(S_i))

    S_E_max = max(entropies)

    # --- Pairwise mutual information ---
    mutual_info = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            p00 = sum(1 for bs in bitstrings if bs[i] == '0' and bs[j] == '0') / n_shots
            p01 = sum(1 for bs in bitstrings if bs[i] == '0' and bs[j] == '1') / n_shots
            p10 = sum(1 for bs in bitstrings if bs[i] == '1' and bs[j] == '0') / n_shots
            p11 = sum(1 for bs in bitstrings if bs[i] == '1' and bs[j] == '1') / n_shots

            probs = [p00, p01, p10, p11]
            S_ij = -sum(p * np.log(max(p, eps)) for p in probs)

            I_ij = entropies[i] + entropies[j] - S_ij
            mutual_info[i, j] = I_ij
            mutual_info[j, i] = I_ij

    return {
        "entropies": entropies,
        "S_E_max": float(S_E_max),
        "mutual_information": mutual_info.tolist(),
        "max_mutual_info": float(np.max(mutual_info)),
        "n_shots": n_shots,
    }


# ============================================================================
# 4. PASQAL CLOUD SUBMISSION
# ============================================================================

def get_client():
    """Authenticate with Pasqal Cloud SDK."""
    if SDK is None:
        raise ImportError("pasqal-cloud not installed. pip install pasqal-cloud")

    project_id = os.environ.get("PASQAL_PROJECT_ID")
    username = os.environ.get("PASQAL_USERNAME")
    password = os.environ.get("PASQAL_PASSWORD")

    if not all([project_id, username, password]):
        print("ERROR: Set PASQAL_PROJECT_ID, PASQAL_USERNAME, PASQAL_PASSWORD")
        sys.exit(1)

    print(f"Authenticating (project: {project_id[:8]}...)...")
    sdk = SDK(project_id=project_id, username=username, password=password)
    print("✓ Authenticated.\n")
    return sdk


def submit_and_collect(sdk, seq, system_name, n_shots=1000,
                       device_type="EMU_FREE", poll_interval=5):
    """Submit sequence to Pasqal Cloud and collect results."""
    serialized = seq.to_abstract_repr()

    print(f"  Submitting {system_name} to {device_type} ({n_shots} shots)...", end=" ")
    batch = sdk.create_batch(
        serialized_sequence=serialized,
        jobs=[{"runs": n_shots}],
        device_type=device_type,
    )
    print(f"batch {batch.id}")

    # Poll until done
    while True:
        batch = sdk.get_batch(batch.id)
        if batch.status not in ("PENDING", "RUNNING"):
            break
        print(".", end="", flush=True)
        time.sleep(poll_interval)

    print(f"  → {batch.status}")

    if batch.status != "DONE":
        return None

    # Extract bitstrings from results
    bitstrings = []
    for job in batch.ordered_jobs:
        if job.status == "DONE" and hasattr(job, "result") and job.result:
            counts = job.result
            for state, count in counts.items():
                bitstrings.extend([state] * count)

    return bitstrings


# ============================================================================
# 5. LOCAL SIMULATION (no cloud needed)
# ============================================================================

def simulate_locally(seq, n_shots=1000):
    """Run sequence on local Pulser QutipEmulator."""
    try:
        from pulser_simulation import QutipEmulator
    except ImportError:
        from pulser.simulation import QutipEmulator  # older pulser versions

    print("  Running local simulation...", end=" ")
    sim = QutipEmulator.from_sequence(seq)
    results = sim.run()
    samples = results.sample_final_state(n_shots)

    # Convert to list of bitstrings
    bitstrings = []
    for state, count in samples.items():
        bitstrings.extend([state] * count)

    print(f"done ({len(bitstrings)} samples)")
    return bitstrings


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="F_bond on Pasqal — Molecular entanglement measurement"
    )
    parser.add_argument(
        "--mode", choices=["local", "cloud"], default="local",
        help="Run locally (QutipEmulator) or submit to Pasqal Cloud"
    )
    parser.add_argument(
        "--emulator", default="EMU_FREE",
        help="Cloud emulator type: EMU_FREE, EMU_MPS, FRESNEL_CAN1"
    )
    parser.add_argument(
        "--shots", type=int, default=1000,
        help="Number of measurement shots per system"
    )
    parser.add_argument(
        "--systems", nargs="*", default=None,
        help="Systems to run (default: all). Options: " +
             ", ".join(SYSTEMS.keys())
    )
    args = parser.parse_args()

    selected = args.systems if args.systems else list(SYSTEMS.keys())
    sdk = get_client() if args.mode == "cloud" else None

    print("=" * 65)
    print("  F_bond × Pasqal — Quantum Entanglement Measurement")
    print(f"  Mode: {args.mode}  |  Shots: {args.shots}  |  Auto-scaling enabled")
    print("=" * 65)

    all_results = []

    for name in selected:
        if name not in SYSTEMS:
            print(f"  ⚠ Unknown system: {name}, skipping")
            continue

        sys_data = SYSTEMS[name]
        n_atoms = len(sys_data["coords_angstrom"])

        print(f"\n{'─'*60}")
        print(f"  {name}: {sys_data['description']}")
        print(f"  Atoms: {n_atoms}  |  Classical F_bond(B): {sys_data['classical_Fbond_B']}")
        print(f"{'─'*60}")

        # Build sequence (auto-scales coordinates)
        try:
            seq, used_scale = build_fbond_sequence(sys_data["coords_angstrom"])
        except Exception as e:
            print(f"  ✗ Sequence build failed: {e}")
            continue

        # Run
        if args.mode == "cloud":
            bitstrings = submit_and_collect(
                sdk, seq, name, n_shots=args.shots,
                device_type=args.emulator
            )
        else:
            bitstrings = simulate_locally(seq, n_shots=args.shots)

        if not bitstrings:
            print("  ✗ No results obtained")
            continue

        # Extract entanglement
        ent = extract_entanglement(bitstrings, n_atoms)

        # Compare with classical
        agreement = (ent["S_E_max"] / sys_data["classical_SE_max"] * 100
                     if sys_data["classical_SE_max"] > 0 else float("nan"))

        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  Results: {name:<32s}║")
        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  Quantum  S_E,max = {ent['S_E_max']:>10.6f}  nats   ║")
        print(f"  ║  Classic  S_E,max = {sys_data['classical_SE_max']:>10.6f}  nats   ║")
        print(f"  ║  Agreement        = {agreement:>10.1f}  %      ║")
        print(f"  ║  Max mutual info  = {ent['max_mutual_info']:>10.6f}         ║")
        print(f"  ╚══════════════════════════════════════════╝")

        result_entry = {
            "system": name,
            "description": sys_data["description"],
            "n_atoms": n_atoms,
            "n_shots": args.shots,
            "mode": args.mode,
            "quantum_SE_max": ent["S_E_max"],
            "classical_SE_max": sys_data["classical_SE_max"],
            "agreement_pct": round(agreement, 1),
            "quantum_entropies": ent["entropies"],
            "max_mutual_info": ent["max_mutual_info"],
            "classical_Fbond_B": sys_data["classical_Fbond_B"],
            "classical_fe": sys_data["classical_fe"],
        }
        all_results.append(result_entry)

    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"fbond_pasqal_results_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved: {outfile}")

    # Summary table
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'System':<22s} {'Q S_E,max':>10s} {'C S_E,max':>10s} {'Agree%':>8s}")
    print(f"  {'─'*52}")
    for r in all_results:
        print(f"  {r['system']:<22s} {r['quantum_SE_max']:>10.6f} "
              f"{r['classical_SE_max']:>10.6f} {r['agreement_pct']:>7.1f}%")
    print()


if __name__ == "__main__":
    main()

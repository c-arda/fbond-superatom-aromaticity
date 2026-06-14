#!/usr/bin/env python3
"""
Stability-checked rerun of Al4(4-) singlet AND triplet, to resolve the
singlet-triplet gap provenance (audit finding 5).

WHY THIS EXISTS
---------------
The production workflow (automated_fbond_workflow.py) runs RHF/UHF with NO
stability analysis, so the SCF can converge to a higher-energy, internally
unstable solution. For Al4(4-) the deposit now shows the symptom directly:

    singlet CCSD in SI Table S4 gap block : -966.72067 Ha   (-> gap -7.5 kcal/mol)
    singlet CCSD deposited / in Table 1   : -966.67914 Ha   (-> gap -33.6 kcal/mol)

Those two singlet solutions differ by 26 kcal/mol. The triplet (4.17 solution,
-966.73263, N_D 4.1676) IS traceable (FINAL_4.17.json -> fbond.occupation_numbers),
so the triplet is fine; the open question is which SINGLET solution is the true
ground state, and therefore what the real S-T gap and the real Table-1 singlet
N_D are.

This script finds the LOWEST internally-stable SCF solution for each spin state
(following instabilities), runs CCSD, and prints N_D + the gap so the correct
numbers can replace Table 1's singlet row, SI Table S4, and the main-text
"7.5 kcal/mol" claim (or confirm them).

HOW TO RUN  (a few minutes; Al4 / def2-SVP / 72 NOs -- do NOT need the cluster,
but follow the workspace scratch rule):
    conda activate fbond-env
    PYSCF_TMPDIR=/data/pyscf_scratch python rerun_al4_st_gap_stability.py

It writes al4_4minus_st_gap_stability_results.json and changes nothing else.
"""
import os, json
import numpy as np

# --- scratch on /data per workspace rule, with a safe fallback ---
_scratch = "/data/pyscf_scratch" if os.path.isdir("/data") else "/tmp/pyscf_scratch"
os.environ.setdefault("PYSCF_TMPDIR", _scratch)
os.makedirs(os.environ["PYSCF_TMPDIR"], exist_ok=True)

from pyscf import gto, scf, cc

OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "al4_4minus_st_gap_stability_results.json")
# D2h rectangular Al4 (antiaromatic), Angstrom; matches FINAL_4.17.json geometry
GEOM   = """Al  -1.3500  -1.1000  0.0000
Al   1.3500  -1.1000  0.0000
Al   1.3500   1.1000  0.0000
Al  -1.3500   1.1000  0.0000"""
BASIS  = "def2-svp"
CHARGE = -4
N_FROZEN = 4          # 1s on each of 4 Al -- matches the production frozen-core rule
HA2KCAL  = 627.509

# deposited reference values, for the user to compare the rerun against
REF = {
    "singlet_deposited_CCSD": -966.67914, "singlet_deposited_N_D": 4.029,
    "singlet_S4_gap_CCSD":    -966.72067,
    "triplet_FINAL_CCSD":     -966.73263, "triplet_FINAL_N_D": 4.1676,
    "gap_S4_kcal": -7.5, "gap_from_deposited_singlet_kcal": -33.6,
}

def s_orb(n):
    p = n / 2.0
    return 0.0 if p <= 0 or p >= 1 else float(-(p*np.log(p) + (1-p)*np.log(1-p)))

def stable_scf(mf, tag, max_cycle=15):
    """Run SCF, then follow internal instabilities to the lowest stable solution.
    Also reports external (e.g. RHF->UHF) stability, which for the closed-shell
    singlet flags whether the true ground state is broken-symmetry/open-shell."""
    mf.kernel()
    for it in range(max_cycle):
        mo_i, mo_e, stable_i, stable_e = mf.stability(internal=True, external=True, return_status=True)
        print(f"  [{tag}] cycle {it}: E = {mf.e_tot:.6f}  internal_stable={stable_i}  external_stable={stable_e}")
        if stable_i:
            if not stable_e:
                print(f"  [{tag}] NOTE: internally stable but EXTERNALLY unstable "
                      f"(e.g. RHF->UHF); the true ground state may be broken-symmetry.")
            return mf, bool(stable_i), bool(stable_e)
        dm = mf.make_rdm1(mo_i, mf.mo_occ)
        mf.kernel(dm0=dm)
    print(f"  [{tag}] WARNING: did not reach internal stability in {max_cycle} cycles.")
    return mf, False, False

def run_state(spin, label):
    mol = gto.M(atom=GEOM, basis=BASIS, charge=CHARGE, spin=spin, verbose=3)
    open_shell = spin > 0
    mf = (scf.UHF if open_shell else scf.RHF)(mol)
    mf, stab_i, stab_e = stable_scf(mf, label)
    e_hf = mf.e_tot
    mycc = (cc.UCCSD if open_shell else cc.CCSD)(mf, frozen=N_FROZEN)
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError(f"{label}: CCSD did not converge")
    e_ccsd = mycc.e_tot
    dm1 = mycc.make_rdm1()
    dm1_tot = (dm1[0] + dm1[1]) if isinstance(dm1, (tuple, list)) else dm1
    occ = np.linalg.eigh(dm1_tot)[0][::-1]
    nd    = float(sum(n*(2-n) for n in occ))
    mfrac = int(sum(1 for n in occ if 1e-3 < n < 2-1e-3))
    return {
        "label": label, "spin": spin,
        "E_HF": float(e_hf), "E_CCSD": float(e_ccsd), "E_corr": float(e_ccsd - e_hf),
        "N_D": nd, "M_frac_1e-3": mfrac, "pct_frac": round(100*mfrac/len(occ)),
        "S_E_max": float(max(s_orb(n) for n in occ)),
        "scf_internally_stable": stab_i, "scf_externally_stable": stab_e,
        "n_orbitals": len(occ),
        "natural_occupations_full": [float(x) for x in occ],
    }

if __name__ == "__main__":
    print("=" * 72)
    print(" Al4(4-) singlet-triplet gap : stability-checked rerun (finding 5)")
    print("=" * 72)
    res = {}
    for spin, label in [(0, "singlet"), (2, "triplet")]:
        print(f"\n--- {label} (spin={spin}) ---")
        res[label] = run_state(spin, label)
        r = res[label]
        print(f"  STABLE: E_HF={r['E_HF']:.5f}  E_CCSD={r['E_CCSD']:.5f}  "
              f"N_D={r['N_D']:.4f}  M_frac={r['M_frac_1e-3']} ({r['pct_frac']}%)")

    gap = (res["triplet"]["E_CCSD"] - res["singlet"]["E_CCSD"]) * HA2KCAL
    res["gap_T_minus_S_kcal"] = float(gap)
    res["_reference"] = REF

    print("\n" + "=" * 72)
    print(f" S-T gap (triplet - singlet) = {gap:.1f} kcal/mol")
    print(f"   reference: SI S4 quotes -7.5 ; deposited singlet would imply -33.6")
    s, t = res["singlet"], res["triplet"]
    print(f" singlet stable CCSD = {s['E_CCSD']:.5f}  (deposit -966.67914 ; S4 gap -966.72067)")
    print(f"   -> N_D = {s['N_D']:.4f}   (Table 1 currently 4.03)")
    print(f" triplet stable CCSD = {t['E_CCSD']:.5f}  (FINAL 4.17 -966.73263)")
    print(f"   -> N_D = {t['N_D']:.4f}   (Table 1 currently 4.17)")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")
    print("ACTION: if the stable singlet differs from -966.67914 / N_D 4.03, then")
    print("Table 1's singlet row, SI Table S4, and the main-text '7.5 kcal/mol'")
    print("all need updating to the stable values printed above.")

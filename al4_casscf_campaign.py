#!/usr/bin/env python3
"""
Phase A: spin-pure CASSCF treatment of the multireference Al4 pair.

Fast production pass: CASSCF over the AVAS 3p-valence active space (~CAS(12,12)),
conventional FCI solver, fully converged, at the CORRECT compact (Bohr) geometry.
This gives the spin-pure natural-orbital N_D + the multireference diagnostic
(active natural occupations near 1.0 = open-shell/diradical; leading-determinant
weight). The slow full-valence (16,17) selected-CI convergence check is a
SEPARATE, optional follow-up (~1-2 h/system) -- not run here.

N_D = sum over active natural occupations n_i (2 - n_i).

Single-reference context (from rerun_al4_compact_geometry):
  Al4(2-)  RHF-CCSD N_D 2.538 ; BS-UCCSD 3.165
  Al4(4-)S RHF-CCSD N_D 2.528 ; BS-UCCSD 3.047
  Al4(4-)T UHF-UCCSD N_D 4.325

RUN (local, ~5-10 min):
    conda activate fbond-env
    PYSCF_TMPDIR=/data/pyscf_scratch python -u al4_casscf_campaign.py
Writes al4_casscf_campaign_results.json incrementally.
"""
import os
# cap threads BEFORE numpy/MKL import -- prevents the OMPxMKL nesting that
# spawned 94 threads and drove load to 52 on the 16C/32T host.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("OMP_NESTED", "FALSE")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
import json, time, traceback
import numpy as np

_scratch = "/data/pyscf_scratch" if os.path.isdir("/data") else "/tmp/pyscf_scratch"
os.environ.setdefault("PYSCF_TMPDIR", _scratch)
os.makedirs(os.environ["PYSCF_TMPDIR"], exist_ok=True)

from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf import avas
lib.num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

HERE  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(HERE, "al4_casscf_campaign_results.json")
BASIS = "def2-svp"
T0    = time.time()
def tnow(): return f"t={time.time()-T0:.0f}s"

# (label, xyz, charge, spin, n_corr_for_fe, single_ref_ccsd_ND, bs_uccsd_ND)
SYSTEMS = [
    ("Al4(2-)",   "Al4_2minus_structure.xyz",         -2, 0, 46, 2.538, 3.165),
    ("Al4(4-) S", "Al4_4minus_structure.xyz",         -4, 0, 48, 2.528, 3.047),
    ("Al4(4-) T", "Al4_4minus_triplet_structure.xyz", -4, 2, 48, 4.325, None),
]

def read_xyz(path):
    lines = open(path).read().strip().splitlines()
    return "\n".join(" ".join(l.split()[:4]) for l in lines[2:] if len(l.split()) >= 4)

def ref_scf(mol, spin):
    mf = scf.ROHF(mol) if spin > 0 else scf.RHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    if spin == 0:
        for _ in range(8):
            try:
                mo_i, mo_e, si, se = mf.stability(internal=True, external=True, return_status=True)
            except Exception:
                break
            if si:
                break
            mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
    return mf

def run_casscf_3p(mf, mol, spin):
    # threshold 0.40 = the validated 3p-valence space (~CAS(12,12), 69s in smoke);
    # 0.10 over-included orbitals and ballooned the FCI -> do NOT lower it.
    ncas, nelecas, mo = avas.avas(mf, ["Al 3p"], threshold=0.40,
                                  openshell_option=(3 if spin > 0 else 2),
                                  canonicalize=True)
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = 100
    mc.conv_tol = 1e-7
    mc.kernel(mo)
    casdm1 = np.asarray(mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas))
    if casdm1.ndim == 3:
        casdm1 = casdm1[0] + casdm1[1]
    occ = np.clip(np.linalg.eigvalsh(casdm1), 0.0, 2.0)
    nd = float(sum(n * (2 - n) for n in occ))
    try:
        w0 = float((np.asarray(mc.ci).ravel() ** 2).max())
    except Exception:
        w0 = None
    return {"ncas": int(ncas),
            "nelecas": list(nelecas) if isinstance(nelecas, (tuple, list)) else int(nelecas),
            "E_CASSCF": float(mc.e_tot), "converged": bool(mc.converged),
            "N_D": nd, "leading_weight": w0,
            "active_noons": [round(float(x), 4) for x in sorted(occ)[::-1]]}

def main():
    results = {}
    print("=" * 92)
    print(" Phase A: Al4 CASSCF over AVAS 3p-valence (spin-pure)  [compact/Bohr geometry]")
    print("=" * 92)
    for label, xyz, charge, spin, ncorr, srccsd, bsccsd in SYSTEMS:
        coords = read_xyz(os.path.join(HERE, "structures", xyz))
        mol = gto.M(atom=coords, unit="Bohr", basis=BASIS, charge=charge, spin=spin, verbose=0)
        mf = ref_scf(mol, spin)
        print(f"\n### {label}  (charge={charge} spin={spin})  E_ref={mf.e_tot:.5f}  {tnow()}", flush=True)
        rec = {"charge": charge, "spin": spin, "n_corr": ncorr,
               "ref_E_HF": float(mf.e_tot),
               "single_ref_CCSD_ND": srccsd, "BS_UCCSD_ND": bsccsd}
        try:
            r = run_casscf_3p(mf, mol, spin)
            r["f_e"] = round(r["N_D"] / ncorr, 4)
            rec["casscf_3p"] = r
            print(f"   CAS({r['nelecas']},{r['ncas']})  E={r['E_CASSCF']:.5f} conv={r['converged']}  "
                  f"N_D={r['N_D']:.4f}  f_e={r['f_e']:.4f}  w0={r['leading_weight']}  {tnow()}", flush=True)
            print(f"   active NOONs (top): {r['active_noons'][:8]}", flush=True)
        except Exception as e:
            rec["casscf_3p"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"   ERROR {type(e).__name__}: {str(e)[:90]}", flush=True)
            traceback.print_exc()
        results[label] = rec
        json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}  {tnow()}")
    print("\nSUMMARY  N_D by treatment (compact geometry):")
    hdr = "  " + "system".ljust(12) + "CCSD(SR)".rjust(10) + "BS-UCCSD".rjust(10) + "CASSCF(3p)".rjust(12) + "f_e(CAS)".rjust(10)
    print(hdr)
    for label, *_ in [(s[0],) for s in SYSTEMS]:
        rec = results.get(label, {})
        c = rec.get("casscf_3p", {})
        sr = rec.get("single_ref_CCSD_ND"); bs = rec.get("BS_UCCSD_ND")
        sr_s = (f"{sr:.3f}" if sr else "-")
        bs_s = (f"{bs:.3f}" if bs else "-")
        nd_s = (f"{c['N_D']:.3f}" if 'N_D' in c else "-")
        fe_s = (f"{c['f_e']:.4f}" if 'f_e' in c else "-")
        print("  " + label.ljust(12) + sr_s.rjust(10) + bs_s.rjust(10) + nd_s.rjust(12) + fe_s.rjust(10))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SCOPE CHECK for audit finding 5.

The production workflow ran RHF/UHF with NO stability analysis. For Al4(4-)
that produced an N_D from a non-stable (and ultimately broken-symmetry) SCF
solution. This script runs SCF + a full stability analysis (internal AND
external) on every manuscript system -- NO CCSD -- to find out which other
production references are unstable, i.e. how big the problem is.

Verdict per system:
  OK               internal+external stable -> single-reference N_D is on solid ground
  EXT-unstable(BS) stable as RHF but wants to break to UHF -> multireference /
                   broken-symmetry; single-reference N_D is suspect
  INT-UNSTABLE     the SCF was not even a local minimum -> N_D unreliable (Al4(4-) case)

RUN (a few-to-many minutes; the 300-AO systems are the slow ones):
    conda activate fbond-env
    PYSCF_TMPDIR=/data/pyscf_scratch python scf_stability_scope.py
"""
import os, json
os.environ.setdefault("PYSCF_TMPDIR", "/data/pyscf_scratch" if os.path.isdir("/data") else "/tmp/pyscf_scratch")
os.makedirs(os.environ["PYSCF_TMPDIR"], exist_ok=True)
from pyscf import gto, scf

HEAVY   = {'Cs','Rb','Ba','Sr','K','Au','Ag','Pt','Hg','Pb','Bi'}   # workflow's ECP set
HERE    = os.path.dirname(os.path.abspath(__file__))
GEOMDIR = os.path.join(HERE, "geometries")
OUT     = os.path.join(HERE, "scf_stability_scope_results.json")

# (label, geometry file, basis override).  B12N12 = the 24-atom 'fulborene' file
# (the 'cage' file is only 12 atoms); Au13 uses its production LANL2DZ basis.
SYSTEMS = [
    ("C6H6",       "C6H6_benzene.json",       None),
    ("Al4(2-)",    "Al4_2minus.json",         None),
    ("Al4(4-) S",  "Al4_4minus.json",         None),
    ("Al4(4-) T",  "Al4_4minus_triplet.json", None),
    ("B12 planar", "B12_planar.json",         None),
    ("B12 ico",    "B12_icosahedral.json",    None),
    ("B6N6",       "B6N6_planar.json",        None),
    ("Cs3Al8",     "Cs3Al8.json",             None),
    ("Cs3Al12",    "Cs3Al12.json",            None),
    ("B12N12",     "B12N12_fulborene.json",   None),
    ("Au13",       "Au13_minus.json",         "lanl2dz"),
]

def build(geomfile, basis_override):
    d = json.load(open(os.path.join(GEOMDIR, geomfile)))
    geom, charge, spin = d["geometry"], d["charge"], d["spin"]
    basis = basis_override or d.get("basis", "def2-svp")
    elements = set(l.split()[0] for l in geom.strip().splitlines() if l.strip())
    basis_dict = {el: basis for el in elements}
    ecp_dict   = {el: basis for el in elements if el in HEAVY}
    mol = gto.M(atom=geom, basis=basis_dict, ecp=(ecp_dict or None),
                charge=charge, spin=spin, verbose=0)
    return mol, spin

def check(label, geomfile, basis_override):
    mol, spin = build(geomfile, basis_override)
    mf = (scf.UHF if spin > 0 else scf.RHF)(mol)
    mf.max_cycle = 200
    mf.kernel()
    e0, conv0 = mf.e_tot, mf.converged
    mo_i, mo_e, stab_i, stab_e = mf.stability(internal=True, external=True, return_status=True)
    e1 = None
    if not stab_i:                                   # follow internal instability once
        mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
        e1 = mf.e_tot
    return {"label": label, "n_ao": int(mol.nao), "nelec": int(mol.nelectron),
            "spin": spin, "E_scf": float(e0), "scf_converged": bool(conv0),
            "internal_stable": bool(stab_i), "external_stable": bool(stab_e),
            "E_after_follow": (float(e1) if e1 is not None else None),
            "dE_lowering_mHa": (float((e1 - e0) * 1000) if e1 is not None else 0.0)}

if __name__ == "__main__":
    print("=" * 90)
    print(" SCOPE CHECK: SCF stability across all manuscript systems (no CCSD)")
    print("=" * 90)
    print(f"{'system':12}{'nAO':>5}{'spin':>5}  {'E_scf':>15}  {'int':>6}{'ext':>6}  {'dE(mHa)':>9}  verdict")
    print("-" * 90)
    results = []
    for label, gf, bo in SYSTEMS:
        try:
            r = check(label, gf, bo)
            results.append(r)
            verdict = ("OK" if (r["internal_stable"] and r["external_stable"])
                       else ("EXT-unstable(BS)" if r["internal_stable"] else "INT-UNSTABLE"))
            print(f"{r['label']:12}{r['n_ao']:>5}{r['spin']:>5}  {r['E_scf']:>15.5f}  "
                  f"{str(r['internal_stable'])[0]:>6}{str(r['external_stable'])[0]:>6}  "
                  f"{r['dE_lowering_mHa']:>9.1f}  {verdict}", flush=True)
        except Exception as e:
            print(f"{label:12}  ERROR: {type(e).__name__}: {str(e)[:60]}", flush=True)
            results.append({"label": label, "error": f"{type(e).__name__}: {e}"})
    json.dump(results, open(OUT, "w"), indent=2)
    bad = [r for r in results if r.get("error") or not r.get("internal_stable", True)
           or not r.get("external_stable", True)]
    print("\nWrote", OUT)
    print(f"\nSUMMARY: {len(results)-len(bad)}/{len(results)} clean; "
          f"{len(bad)} need attention -> " + ", ".join(r['label'] for r in bad))

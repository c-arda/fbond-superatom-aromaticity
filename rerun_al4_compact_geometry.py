#!/usr/bin/env python3
"""
Al4 family: geometry-units + SCF-stability correction (audit finding 5, extended).

TWO compounding problems were found in the deposited Al4 data:

  (1) GEOMETRY UNITS. structures/Al4_*_structure.xyz and geometries/Al4_*.json
      store coordinates in BOHR but in .xyz format (no unit field). The
      production pipeline read them as Angstrom, so every Al4 system was
      computed at a geometry ~1.89x too large (near-dissociated):
          Al4(2-) square edge  4.724315  read as A  = 4.72 A   (true 2.50 A)
          Al4(4-) short edge   4.157398  read as A  = 4.16 A   (true 2.20 A)
      Only the 3 Al4 systems are Bohr-stored; benzene/B12/B6N6/B12N12/Cs3Al8/
      Cs3Al12/Au13 are genuine Angstrom and are unaffected.

  (2) SCF STABILITY. Even at the correct geometry the Al4(4-) singlet RHF is
      internally unstable and, once internally stable, still RHF->UHF
      (externally) unstable -> multireference / broken-symmetry. Single-
      reference CCSD on the unstable reference is not valid.

This script, for each Al4 system, at the CORRECT (Bohr) geometry:
  - confirms the deposit by also building the STRETCHED (read-as-Angstrom)
    geometry and printing its E_HF (should match the deposited E_HF);
  - finds the lowest internally-stable RHF/UHF reference (follows instabilities);
  - for the closed-shell singlets, also seeks the broken-symmetry UHF solution;
  - runs CCSD/UCCSD and reports N_D, f_e, M_frac, <S^2> for each.

It writes al4_compact_geometry_results.json and changes nothing else.

RUN (a few minutes, def2-SVP / 72 AO):
    conda activate fbond-env
    PYSCF_TMPDIR=/data/pyscf_scratch python rerun_al4_compact_geometry.py
"""
import os, json
import numpy as np

_scratch = "/data/pyscf_scratch" if os.path.isdir("/data") else "/tmp/pyscf_scratch"
os.environ.setdefault("PYSCF_TMPDIR", _scratch)
os.makedirs(os.environ["PYSCF_TMPDIR"], exist_ok=True)

from pyscf import gto, scf, cc

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, "al4_compact_geometry_results.json")
BASIS  = "def2-svp"
N_FROZEN = 4          # 1s on each Al; inert, so N_D is insensitive to this
HA2KCAL  = 627.509

# (label, xyz file, charge, spin, deposited N_D, deposited E_HF, deposited E_CCSD, n_corr)
SYSTEMS = [
    ("Al4(2-)",   "Al4_2minus_structure.xyz",          -2, 0, 3.840, -966.95638, -967.25671, 46),
    ("Al4(4-) S", "Al4_4minus_structure.xyz",          -4, 0, 4.029, -966.32693, -966.67914, 48),
    ("Al4(4-) T", "Al4_4minus_triplet_structure.xyz",  -4, 2, 4.168, -966.39038, -966.73263, 48),
]

def read_xyz_coords(path):
    """Return the atom block string (element x y z per line) from an .xyz file."""
    lines = open(path).read().strip().splitlines()
    body = []
    for ln in lines[2:]:
        p = ln.split()
        if len(p) >= 4:
            body.append(f"{p[0]} {p[1]} {p[2]} {p[3]}")
    return "\n".join(body)

def s_orb(n):
    p = n / 2.0
    return 0.0 if p <= 0 or p >= 1 else float(-(p*np.log(p) + (1-p)*np.log(1-p)))

def noons_nd(mycc):
    dm1 = mycc.make_rdm1()
    dm1_tot = (dm1[0] + dm1[1]) if isinstance(dm1, (tuple, list)) else dm1
    occ = np.linalg.eigh(dm1_tot)[0][::-1]
    nd  = float(sum(n*(2-n) for n in occ))
    return occ, nd

def follow_internal(mf, tag, max_cycle=20):
    """Run SCF then follow internal instabilities to the lowest stable solution."""
    mf.kernel()
    for it in range(max_cycle):
        mo_i, mo_e, si, se = mf.stability(internal=True, external=True, return_status=True)
        if si:
            return mf, bool(si), bool(se)
        mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
    return mf, False, None

def spin_sq(mf):
    try:
        ss = mf.spin_square()[0]
        return float(ss)
    except Exception:
        return None

def run_ref(mol, kind, tag):
    """kind in {'R','U','BS'}; returns dict of reference + CCSD results, or None."""
    if kind == "R":
        mf = scf.RHF(mol); mf.max_cycle = 200
        mf, si, se = follow_internal(mf, tag)
    elif kind == "U":
        mf = scf.UHF(mol); mf.max_cycle = 200
        mf, si, se = follow_internal(mf, tag)
    elif kind == "BS":
        # broken-symmetry singlet: start UHF from default guess; the symmetric
        # (RHF-like) UHF solution is internally unstable when RHF is RHF->UHF
        # unstable, so following internal instability breaks spin symmetry.
        mf = scf.UHF(mol); mf.max_cycle = 200
        mf, si, se = follow_internal(mf, tag)
    else:
        raise ValueError(kind)

    open_shell = isinstance(mf, scf.uhf.UHF)
    e_hf = float(mf.e_tot)
    ss   = spin_sq(mf)
    mycc = (cc.UCCSD if open_shell else cc.CCSD)(mf, frozen=N_FROZEN)
    mycc.max_cycle = 150
    mycc.kernel()
    if not mycc.converged:
        return {"kind": kind, "E_HF": e_hf, "S2": ss, "int_stable": si,
                "ext_stable": se, "ccsd_converged": False}
    occ, nd = noons_nd(mycc)
    mfrac = int(sum(1 for n in occ if 1e-3 < n < 2 - 1e-3))
    return {"kind": kind, "E_HF": e_hf, "E_CCSD": float(mycc.e_tot),
            "E_corr": float(mycc.e_tot - e_hf), "N_D": nd,
            "M_frac": mfrac, "S_E_max": float(max(s_orb(n) for n in occ)),
            "S2": ss, "int_stable": bool(si),
            "ext_stable": (None if se is None else bool(se)),
            "ccsd_converged": True,
            "noons_top": [float(x) for x in occ[:8]]}

def main():
    print("=" * 100)
    print(" Al4 family: correct (compact/Bohr) geometry + stability-checked CCSD")
    print("=" * 100)
    results = {}
    for label, xyz, charge, spin, dep_nd, dep_ehf, dep_eccsd, ncorr in SYSTEMS:
        coords = read_xyz_coords(os.path.join(HERE, "structures", xyz))
        print(f"\n{'='*100}\n### {label}   charge={charge} spin={spin}   "
              f"deposit: N_D={dep_nd}  E_HF={dep_ehf}  E_CCSD={dep_eccsd}  n_corr={ncorr}")

        # --- (1) reproduce the deposit: STRETCHED geometry (read Bohr values as Angstrom) ---
        mol_str = gto.M(atom=coords, unit="Angstrom", basis=BASIS,
                        charge=charge, spin=spin, verbose=0)
        mf_str = (scf.UHF if spin > 0 else scf.RHF)(mol_str); mf_str.max_cycle = 200
        mf_str.kernel()
        e_str = float(mf_str.e_tot)
        match = "MATCHES deposit (deposit was computed STRETCHED)" if abs(e_str - dep_ehf) < 2e-3 \
                else "does NOT match deposit"
        nn_str = mol_str.atom_coords()  # bohr internally
        print(f"  [stretched/read-as-A] min Al-Al ~ "
              f"{_mindist_ang(mol_str):.2f} A   E_HF={e_str:.5f}   -> {match}")

        # --- (2) correct COMPACT geometry (Bohr) ---
        mol = gto.M(atom=coords, unit="Bohr", basis=BASIS,
                    charge=charge, spin=spin, verbose=0)
        print(f"  [compact/correct ]   min Al-Al ~ {_mindist_ang(mol):.2f} A")

        sysres = {"label": label, "charge": charge, "spin": spin,
                  "deposit": {"N_D": dep_nd, "E_HF": dep_ehf, "E_CCSD": dep_eccsd,
                              "n_corr": ncorr, "geometry": "stretched (read-as-Angstrom)"},
                  "stretched_check": {"E_HF": e_str, "matches_deposit": abs(e_str-dep_ehf)<2e-3},
                  "compact": {}}

        if spin == 0:
            # restricted (stable-internal) and broken-symmetry
            for kind in ("R", "BS"):
                tag = f"{label}/{kind}"
                r = run_ref(mol, kind, tag)
                r["f_e"] = (r["N_D"]/ncorr) if r.get("N_D") is not None else None
                sysres["compact"][kind] = r
                _print_ref(label, r)
        else:
            r = run_ref(mol, "U", f"{label}/U")
            r["f_e"] = (r["N_D"]/ncorr) if r.get("N_D") is not None else None
            sysres["compact"]["U"] = r
            _print_ref(label, r)

        results[label] = sysres

    # --- corrected S-T gap from compact, stable references ---
    try:
        s = results["Al4(4-) S"]["compact"]
        sing = s.get("BS") if (s.get("BS", {}).get("ccsd_converged") and
                               s["BS"]["E_CCSD"] <= s["R"]["E_CCSD"]) else s["R"]
        trip = results["Al4(4-) T"]["compact"]["U"]
        gap = (trip["E_CCSD"] - sing["E_CCSD"]) * HA2KCAL
        results["_gap_T_minus_S_kcal_compact"] = float(gap)
        print(f"\n{'='*100}\n CORRECTED S-T gap (compact geometry, stable refs): "
              f"{gap:+.1f} kcal/mol   [singlet ref = {sing['kind']}]")
    except Exception as e:
        print("gap calc skipped:", e)

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")

def _mindist_ang(mol):
    import itertools
    c = mol.atom_coords() * 0.52917721092  # bohr -> angstrom
    return min(np.linalg.norm(c[i]-c[j]) for i, j in itertools.combinations(range(len(c)), 2))

def _print_ref(label, r):
    if not r.get("ccsd_converged"):
        print(f"    [{r['kind']:2}] E_HF={r['E_HF']:.5f}  CCSD DID NOT CONVERGE  "
              f"int={r['int_stable']} ext={r['ext_stable']} <S2>={r['S2']}")
        return
    print(f"    [{r['kind']:2}] E_HF={r['E_HF']:.5f}  E_CCSD={r['E_CCSD']:.5f}  "
          f"N_D={r['N_D']:.4f}  f_e={r['f_e']:.4f}  M_frac={r['M_frac']}  "
          f"<S2>={r['S2']:.3f}  int={r['int_stable']} ext={r['ext_stable']}")

if __name__ == "__main__":
    main()

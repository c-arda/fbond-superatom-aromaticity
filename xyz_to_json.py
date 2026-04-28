#!/usr/bin/env python3
"""
Convert XYZ structure files to batch-dispatchable geometry JSON files.

Usage:
    python xyz_to_json.py                    # Convert all structures/*.xyz
    python xyz_to_json.py structures/C6H6_benzene_structure.xyz
"""
import json
import os
import sys
from pathlib import Path

# ============================================================================
# System metadata — charge, spin, notes
# Sources: fbond_pasqal.py, basis_set_validation scripts, literature
# ============================================================================
SYSTEM_METADATA = {
    "Al4_2minus": {
        "charge": -2,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "D4h square planar, aromatic (2π e⁻). Li & Boldyrev 2003."
    },
    "Al4_4minus": {
        "charge": -4,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "D2h rectangular, antiaromatic singlet (4π e⁻)."
    },
    "Al4_4minus_triplet": {
        "charge": -4,
        "spin": 2,
        "basis": "def2-SVP",
        "notes": "D2h rectangular, antiaromatic triplet state."
    },
    "Au13_minus": {
        "charge": -1,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "Au13⁻ icosahedral cluster. Requires def2-ECP for Au."
    },
    "B12_icosahedral": {
        "charge": 0,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "B12 icosahedral cage, Wade's rules."
    },
    "B12N12_cage": {
        "charge": 0,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "B12N12 cage (fulborene). Note: XYZ may list only 12 atoms if BN alternating."
    },
    "B12_planar": {
        "charge": 0,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "D3h planar B12, quasi-planar boron sheet."
    },
    "B6N6_planar": {
        "charge": 0,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "Borazine-like B6N6 ring, isoelectronic to benzene."
    },
    "C6H6_benzene": {
        "charge": 0,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "Reference aromatic: benzene (D6h)."
    },
    "Cs3Al8": {
        "charge": -1,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "Superatom cluster, 40-electron shell closure."
    },
    "Cs3Al12": {
        "charge": -1,
        "spin": 0,
        "basis": "def2-SVP",
        "notes": "Larger superatom cluster."
    },
}


def xyz_to_json(xyz_path, output_dir="geometries"):
    """Convert a single XYZ file to geometry JSON."""
    xyz_path = Path(xyz_path)
    if not xyz_path.exists():
        print(f"  ✗ Not found: {xyz_path}")
        return None

    with open(xyz_path) as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    geometry_lines = [l.strip() for l in lines[2:2 + n_atoms] if l.strip()]

    # Derive system name from filename: structures/Al4_2minus_structure.xyz → Al4_2minus
    stem = xyz_path.stem  # e.g. "Al4_2minus_structure"
    name = stem.replace("_structure", "")

    # Look up metadata
    meta = SYSTEM_METADATA.get(name)
    if meta is None:
        print(f"  ⚠ No metadata for '{name}', using defaults (charge=0, spin=0)")
        meta = {"charge": 0, "spin": 0, "basis": "def2-SVP", "notes": ""}

    geometry = "\n".join(geometry_lines)

    data = {
        "name": name,
        "geometry": geometry,
        "charge": meta["charge"],
        "spin": meta["spin"],
        "basis": meta["basis"],
        "notes": meta.get("notes", ""),
        "source_xyz": str(xyz_path),
        "source_comment": comment,
        "n_atoms": n_atoms,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  ✓ {name:25s} → {out_path}  (charge={meta['charge']}, spin={meta['spin']}, atoms={n_atoms})")
    return out_path


def main():
    if len(sys.argv) > 1:
        # Convert specific files
        for path in sys.argv[1:]:
            xyz_to_json(path)
    else:
        # Convert all structures/*.xyz
        struct_dir = Path(__file__).parent / "structures"
        if not struct_dir.exists():
            print(f"ERROR: {struct_dir} not found")
            sys.exit(1)

        xyz_files = sorted(struct_dir.glob("*.xyz"))
        print(f"Converting {len(xyz_files)} XYZ files → geometry JSONs\n")

        converted = []
        for xyz in xyz_files:
            result = xyz_to_json(xyz)
            if result:
                converted.append(result)

        print(f"\n✓ {len(converted)}/{len(xyz_files)} files converted to geometries/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cube File Orbital Visualizer
Reads Gaussian/PySCF cube files and creates 3D isosurface visualization
"""
import numpy as np
import plotly.graph_objects as go
from skimage import measure
import sys

def read_cube_file(filename):
    """Parse a cube file and return geometry and volumetric data"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip first two comment lines
    natoms = int(lines[2].split()[0])
    origin = np.array([float(x) for x in lines[2].split()[1:4]])

    # Read grid dimensions
    nx = int(lines[3].split()[0])
    vx = np.array([float(x) for x in lines[3].split()[1:4]])

    ny = int(lines[4].split()[0])
    vy = np.array([float(x) for x in lines[4].split()[1:4]])

    nz = int(lines[5].split()[0])
    vz = np.array([float(x) for x in lines[5].split()[1:4]])

    # Read atomic coordinates
    atoms = []
    for i in range(6, 6 + natoms):
        parts = lines[i].split()
        atom_num = int(parts[0])
        charge = float(parts[1])
        coords = np.array([float(x) for x in parts[2:5]])
        atoms.append((atom_num, coords))

    # Read volumetric data
    data_start = 6 + natoms
    data_lines = lines[data_start:]
    data_text = ' '.join(data_lines)
    data = np.array([float(x) for x in data_text.split()])

    # Reshape to 3D grid
    data = data.reshape((nx, ny, nz))

    # Create coordinate grids
    x = origin[0] + np.arange(nx) * vx[0]
    y = origin[1] + np.arange(ny) * vy[1]
    z = origin[2] + np.arange(nz) * vz[2]

    return atoms, (x, y, z), data

def visualize_orbital(cube_file, isovalue=0.03, output_html='orbital_view.html'):
    """Create interactive 3D visualization of molecular orbital"""
    print(f"Reading {cube_file}...")
    atoms, (x, y, z), data = read_cube_file(cube_file)

    print(f"Grid dimensions: {data.shape}")
    print(f"Data range: {data.min():.6f} to {data.max():.6f}")
    print(f"Using isovalue: ±{isovalue}")

    # Convert Bohr to Angstrom for display
    BOHR_TO_ANG = 0.529177
    x_ang = x * BOHR_TO_ANG
    y_ang = y * BOHR_TO_ANG
    z_ang = z * BOHR_TO_ANG
    atoms_ang = [(num, coords * BOHR_TO_ANG) for num, coords in atoms]

    # Create figure
    fig = go.Figure()

    # Add positive isosurface (red)
    try:
        verts_pos, faces_pos, _, _ = measure.marching_cubes(data, level=isovalue)
        # Scale vertices to actual coordinates
        verts_pos_scaled = np.zeros_like(verts_pos)
        verts_pos_scaled[:, 0] = x_ang[verts_pos[:, 0].astype(int)]
        verts_pos_scaled[:, 1] = y_ang[verts_pos[:, 1].astype(int)]
        verts_pos_scaled[:, 2] = z_ang[verts_pos[:, 2].astype(int)]

        fig.add_trace(go.Mesh3d(
            x=verts_pos_scaled[:, 0],
            y=verts_pos_scaled[:, 1],
            z=verts_pos_scaled[:, 2],
            i=faces_pos[:, 0],
            j=faces_pos[:, 1],
            k=faces_pos[:, 2],
            color='red',
            opacity=0.7,
            name='Positive lobe'
        ))
        print(f"✓ Positive isosurface created ({len(verts_pos)} vertices)")
    except Exception as e:
        print(f"Warning: Could not create positive isosurface: {e}")

    # Add negative isosurface (blue)
    try:
        verts_neg, faces_neg, _, _ = measure.marching_cubes(data, level=-isovalue)
        verts_neg_scaled = np.zeros_like(verts_neg)
        verts_neg_scaled[:, 0] = x_ang[verts_neg[:, 0].astype(int)]
        verts_neg_scaled[:, 1] = y_ang[verts_neg[:, 1].astype(int)]
        verts_neg_scaled[:, 2] = z_ang[verts_neg[:, 2].astype(int)]

        fig.add_trace(go.Mesh3d(
            x=verts_neg_scaled[:, 0],
            y=verts_neg_scaled[:, 1],
            z=verts_neg_scaled[:, 2],
            i=faces_neg[:, 0],
            j=faces_neg[:, 1],
            k=faces_neg[:, 2],
            color='blue',
            opacity=0.7,
            name='Negative lobe'
        ))
        print(f"✓ Negative isosurface created ({len(verts_neg)} vertices)")
    except Exception as e:
        print(f"Warning: Could not create negative isosurface: {e}")

    # Add atoms as spheres
    atom_names = {13: 'Al', 55: 'Cs'}
    atom_colors = {13: 'silver', 55: 'gold'}
    atom_sizes = {13: 8, 55: 12}

    for atom_num, coords in atoms_ang:
        fig.add_trace(go.Scatter3d(
            x=[coords[0]],
            y=[coords[1]],
            z=[coords[2]],
            mode='markers',
            marker=dict(
                size=atom_sizes.get(atom_num, 6),
                color=atom_colors.get(atom_num, 'gray'),
                line=dict(width=1, color='black')
            ),
            name=atom_names.get(atom_num, f'Atom {atom_num}'),
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title=f"Molecular Orbital: {cube_file}",
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )

    # Save HTML file
    fig.write_html(output_html)
    print(f"\n✓ Visualization saved to: {output_html}")
    print(f"  Open in browser to view interactive 3D orbital")

    return fig

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_cube.py <cube_file> [isovalue] [output.html]")
        print("\nExample:")
        print("  python visualize_cube.py Cs3Al8_HOMO_NO.cube 0.03 Cs3Al8_HOMO.html")
        sys.exit(1)

    cube_file = sys.argv[1]
    isovalue = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    output_html = sys.argv[3] if len(sys.argv) > 3 else cube_file.replace('.cube', '.html')

    visualize_orbital(cube_file, isovalue, output_html)

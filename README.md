# 3DMRIRendering

## STL Size Optimization

This project keeps the CUDA Marching Cubes reconstruction path intact and applies export-time mesh optimization only.

### Implemented

- Binary STL export
- Pre-export STL size estimation
- Optional QEM mesh simplification during export

### OpenMesh Setup

Install OpenMesh and set `OPENMESH_DIR` before building with qmake.

Example:

```powershell
$env:OPENMESH_DIR = "C:\OpenMesh"
```

Expected layout:

- `C:\OpenMesh\include`
- `C:\OpenMesh\lib`

### Visual Studio / Qt Creator

1. Open the `.pro` file in Qt Creator or run `qmake` from a Qt command prompt.
2. Make sure `OPENMESH_DIR`, CUDA, and OpenCV paths point to your local install.
3. Build the project in Debug or Release mode.

### Export Workflow

1. Generate the mesh normally.
2. Use **Export STL**.
3. Review the estimated size and triangle count.
4. Optionally enable simplification and choose a preset:
	- Ultra: keep full detail
	- Medium: keep 65% of triangles
	- Low: keep 35% of triangles
5. Export the optimized binary STL.
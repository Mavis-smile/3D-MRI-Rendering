# 3D MRI Rendering — First-Time Setup Guide

A Qt/CUDA application for 3D reconstruction and visualization of MRI/CT DICOM datasets using GPU-accelerated Marching Cubes.

---

## 1. Prerequisites

Install **all** of the following before opening the project.

| Dependency | Required Version | Notes |
|---|---|---|
| **Qt** | 6.6.2 | Install via Qt Online Installer |
| **Qt Kit** | MSVC 2019 64-bit | Select during Qt installation |
| **Visual Studio** | 2019 (any edition) | Required for MSVC compiler and Windows SDK |
| **CUDA Toolkit** | 13.1 | From NVIDIA Developer site |
| **NVIDIA GPU** | Compute Capability ≥ 7.5 | e.g. GTX 1650, RTX 20/30/40 series |
| **OpenCV** | 4.12.0 (vc16 x64) | Pre-built binaries for VS2019 |
| **OpenMesh** *(optional)* | Any recent release | Required only for QEM mesh simplification on export |

> **Visual Studio 2019** must be installed even if you only build from Qt Creator, because qmake uses the MSVC toolchain (`vc16`).

---

## 2. Install OpenCV

1. Download the OpenCV 4.12.0 Windows release from [opencv.org](https://opencv.org/releases/).
2. Run the self-extracting archive and extract it to a folder of your choice, for example:
   ```
   C:\opencv
   ```
3. The layout must be:
   ```
   C:\opencv\build\include\
   C:\opencv\build\x64\vc16\lib\
   C:\opencv\build\x64\vc16\bin\
   ```
4. Add the `bin` folder to your system `PATH` so the DLLs are found at runtime:
   ```
   C:\opencv\build\x64\vc16\bin
   ```

---

## 3. Install CUDA Toolkit

1. Download **CUDA Toolkit 13.1** from the [NVIDIA CUDA Archive](https://developer.nvidia.com/cuda-toolkit-archive).
2. Run the installer and use the default path:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
   ```
   If you install to a different path, update `CUDA_PATH` in `3DMRIRendering.pro` (see Section 5).

---

## 4. Install OpenMesh (Optional)

OpenMesh is only required if you want QEM mesh simplification during STL export. The project builds and runs without it.

1. Download OpenMesh from [graphics.rwth-aachen.de](https://www.graphics.rwth-aachen.de/software/openmesh/).
2. Build or install it so the layout is:
   ```
   C:\OpenMesh\include\
   C:\OpenMesh\lib\
   ```
3. Set the `OPENMESH_DIR` environment variable before launching Qt Creator:
   ```powershell
   $env:OPENMESH_DIR = "C:\OpenMesh"
   ```
   Or set it permanently in **System Properties → Environment Variables**.

---

## 5. Edit `3DMRIRendering.pro` for Your Machine

Open `3DMRIRendering.pro` in a text editor and update the following lines to match your local paths.

### 5a. CUDA path and GPU architecture

```qmake
# Line ~19 — change version folder if your CUDA install differs
CUDA_PATH = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1

# Line ~21 — change sm_75 to match your GPU's compute capability
# sm_75 = GTX 1650 / RTX 20xx
# sm_86 = RTX 30xx
# sm_89 = RTX 40xx
CUDA_GENCODE = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
```

To find your GPU's compute capability: [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).

### 5b. OpenCV include path (appears twice — in NVCC commands and in INCLUDEPATH)

```qmake
# Replace C:/Users/weiti/Desktop/opencv with your actual OpenCV root
INCLUDEPATH += C:/Users/weiti/Desktop/opencv/build/include
```

Also update the two NVCC `-I` flags inside the `cuda.commands` blocks (one for debug, one for release):

```qmake
-I\"C:/Users/weiti/Desktop/opencv/build/include\" \
```

### 5c. OpenCV library path

```qmake
# Release
LIBS += -LC:/Users/weiti/Desktop/opencv/build/x64/vc16/lib -lopencv_world4120

# Debug
LIBS += -LC:/Users/weiti/Desktop/opencv/build/x64/vc16/lib -lopencv_world4120d
```

Replace `C:/Users/weiti/Desktop/opencv` with the root folder where you extracted OpenCV.

---

## 6. Configure Qt Creator

1. Open **Qt Creator**.
2. Go to **File → Open File or Project** and select `3DMRIRendering.pro`.
3. Qt Creator will ask you to **configure the project** — select the kit:
   - `Desktop Qt 6.6.2 MSVC2019 64bit`
4. Qt Creator generates a `3DMRIRendering.pro.user` file automatically. Do **not** commit this file.
5. In **Projects → Build Settings**, verify:
   - **Build directory** points to a local writable path.
   - Both **Debug** and **Release** configurations are present.

---

## 7. Build the Project

```powershell
# Option A: from Qt Creator
# Click Build → Build Project (Ctrl+B)

# Option B: from a Qt 6.6.2 MSVC command prompt
cd C:\Users\<you>\Desktop\3DMRIRendering
qmake 3DMRIRendering.pro
nmake      
```

The CUDA source `MarchingCubes.cu` is compiled automatically by `nvcc` as an extra compiler step defined in the `.pro` file.

---

## 8. Runtime DLLs

Make sure the following directories are on your system `PATH` (or copy the DLLs next to the `.exe`):

| Library | DLL location |
|---|---|
| OpenCV (Release) | `<opencv_root>\build\x64\vc16\bin` |
| CUDA Runtime | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin` |
| Qt 6.6.2 | `C:\Qt\6.6.2\msvc2019_64\bin` |

After building in Qt Creator the Qt DLLs are usually deployed automatically to the output folder.

---

## 9. First Run

1. Launch the application (`3DMRIRendering.exe`).
2. A **login dialog** will appear. Use the default credentials or create a new user via the user management panel.
3. Load a DICOM/image dataset using the **Import** button.
4. Adjust the **Threshold** slider (or click **Auto** for Otsu estimation).
5. Click **Generate 3D** to run GPU Marching Cubes and produce the mesh.
6. Optionally enable **Material Colors** to visualize bone/scaffold classification.
7. Use **Export STL** to save the mesh. Choose a simplification preset if needed.
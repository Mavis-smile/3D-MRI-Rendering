#ifndef MESHSIMPLIFIER_H
#define MESHSIMPLIFIER_H

#include "MarchingCubes.h"
#include <QString>

namespace MeshSimplifier {

struct SimplifyReport {
    MarchingCubes::Mesh mesh;
    bool success = false;
    QString message;
    int inputFaceCount = 0;
    int outputFaceCount = 0;
};

bool isOpenMeshAvailable();
QString backendName();
SimplifyReport simplifyMeshDetailed(const MarchingCubes::Mesh& inputMesh, int targetFaceCount, double aggressiveness);
MarchingCubes::Mesh simplifyMesh(const MarchingCubes::Mesh& inputMesh, int targetFaceCount, double aggressiveness);

} // namespace MeshSimplifier

#endif // MESHSIMPLIFIER_H
#include "MeshSimplifier.h"

#include <QDebug>
#include <QElapsedTimer>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#if __has_include(<OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>) && __has_include(<OpenMesh/Tools/Decimater/DecimaterT.hh>) && __has_include(<OpenMesh/Tools/Decimater/ModQuadricT.hh>)
#   define MESH_SIMPLIFIER_HAS_OPENMESH 1
#   include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#   include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#   include <OpenMesh/Tools/Decimater/ModQuadricT.hh>
#else
#   define MESH_SIMPLIFIER_HAS_OPENMESH 0
#endif

namespace MeshSimplifier {
namespace {

#if MESH_SIMPLIFIER_HAS_OPENMESH
struct MeshTraits : public OpenMesh::DefaultTraits {
    VertexAttributes(OpenMesh::Attributes::Status);
    FaceAttributes(OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);
};

using OpenMeshType = OpenMesh::TriMesh_ArrayKernelT<MeshTraits>;
#endif
// (O(N²) winding-number classification removed — it blocked the thread for minutes on large meshes;
//  OpenMesh QEM decimation preserves surface quality without needing a pre-pass.)

struct CleanupStats {
    int weldedVertexCount = 0;
    int removedDuplicateVertices = 0;
    int removedDegenerateFaces = 0;
    int removedDuplicateFaces = 0;
    int removedSmallComponentFaces = 0;
    int componentFaceThreshold = 0;
    int keptComponents = 0;
    bool componentFilteringApplied = false;
    float largestComponentRatio = 1.0f;
};

struct VertexKey {
    quint32 x = 0;
    quint32 y = 0;
    quint32 z = 0;

    bool operator==(const VertexKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VertexKeyHash {
    std::size_t operator()(const VertexKey& key) const {
        const std::size_t h1 = std::hash<quint32>{}(key.x);
        const std::size_t h2 = std::hash<quint32>{}(key.y);
        const std::size_t h3 = std::hash<quint32>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct FaceKey {
    unsigned int a = 0;
    unsigned int b = 0;
    unsigned int c = 0;

    bool operator==(const FaceKey& other) const {
        return a == other.a && b == other.b && c == other.c;
    }
};

struct FaceKeyHash {
    std::size_t operator()(const FaceKey& key) const {
        const std::size_t h1 = std::hash<unsigned int>{}(key.a);
        const std::size_t h2 = std::hash<unsigned int>{}(key.b);
        const std::size_t h3 = std::hash<unsigned int>{}(key.c);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct EdgeKey {
    unsigned int a = 0;
    unsigned int b = 0;

    bool operator==(const EdgeKey& other) const {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const {
        const std::size_t h1 = std::hash<unsigned int>{}(key.a);
        const std::size_t h2 = std::hash<unsigned int>{}(key.b);
        return h1 ^ (h2 << 1);
    }
};

struct TriangleIndices {
    unsigned int a = 0;
    unsigned int b = 0;
    unsigned int c = 0;
};

struct DisjointSet {
    QVector<int> parent;
    QVector<int> size;

    explicit DisjointSet(int count)
        : parent(count), size(count, 1)
    {
        for (int i = 0; i < count; ++i) {
            parent[i] = i;
        }
    }

    int find(int value)
    {
        if (parent[value] != value) {
            parent[value] = find(parent[value]);
        }
        return parent[value];
    }

    void unite(int left, int right)
    {
        left = find(left);
        right = find(right);
        if (left == right) {
            return;
        }
        if (size[left] < size[right]) {
            std::swap(left, right);
        }
        parent[right] = left;
        size[left] += size[right];
    }
};

quint32 floatBits(float value)
{
    quint32 bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

VertexKey makeVertexKey(const QVector3D& vertex)
{
    return VertexKey{floatBits(vertex.x()), floatBits(vertex.y()), floatBits(vertex.z())};
}

FaceKey makeFaceKey(unsigned int a, unsigned int b, unsigned int c)
{
    if (a > b) {
        std::swap(a, b);
    }
    if (b > c) {
        std::swap(b, c);
    }
    if (a > b) {
        std::swap(a, b);
    }
    return FaceKey{a, b, c};
}

EdgeKey makeEdgeKey(unsigned int a, unsigned int b)
{
    if (a > b) {
        std::swap(a, b);
    }
    return EdgeKey{a, b};
}

MarchingCubes::Mesh cleanupMeshTopology(const MarchingCubes::Mesh& inputMesh, CleanupStats* statsOut = nullptr)
{
    CleanupStats stats;
    MarchingCubes::Mesh cleanedMesh;

    if (inputMesh.vertices.isEmpty() || inputMesh.indices.size() < 3) {
        if (statsOut) {
            *statsOut = stats;
        }
        return inputMesh;
    }

    QVector<int> vertexRemap(inputMesh.vertices.size(), -1);
    QVector<QVector3D> weldedVertices;
    weldedVertices.reserve(inputMesh.vertices.size());

    std::unordered_map<VertexKey, unsigned int, VertexKeyHash> vertexMap;
    vertexMap.reserve(static_cast<std::size_t>(inputMesh.vertices.size()));

    for (int i = 0; i < inputMesh.vertices.size(); ++i) {
        const QVector3D& vertex = inputMesh.vertices[i];
        const VertexKey key = makeVertexKey(vertex);
        const auto it = vertexMap.find(key);
        if (it != vertexMap.end()) {
            vertexRemap[i] = static_cast<int>(it->second);
            ++stats.removedDuplicateVertices;
        } else {
            const unsigned int newIndex = static_cast<unsigned int>(weldedVertices.size());
            vertexMap.emplace(key, newIndex);
            vertexRemap[i] = static_cast<int>(newIndex);
            weldedVertices.append(vertex);
        }
    }

    QVector<TriangleIndices> faces;
    faces.reserve(inputMesh.indices.size() / 3);

    std::unordered_map<FaceKey, int, FaceKeyHash> faceMap;
    faceMap.reserve(static_cast<std::size_t>(inputMesh.indices.size() / 3));

    for (int i = 0; i + 2 < inputMesh.indices.size(); i += 3) {
        const unsigned int ia = inputMesh.indices[i + 0];
        const unsigned int ib = inputMesh.indices[i + 1];
        const unsigned int ic = inputMesh.indices[i + 2];
        if (ia >= static_cast<unsigned int>(vertexRemap.size())
            || ib >= static_cast<unsigned int>(vertexRemap.size())
            || ic >= static_cast<unsigned int>(vertexRemap.size())) {
            continue;
        }

        const unsigned int a = static_cast<unsigned int>(vertexRemap[static_cast<int>(ia)]);
        const unsigned int b = static_cast<unsigned int>(vertexRemap[static_cast<int>(ib)]);
        const unsigned int c = static_cast<unsigned int>(vertexRemap[static_cast<int>(ic)]);
        if (a == b || b == c || a == c) {
            ++stats.removedDegenerateFaces;
            continue;
        }

        const QVector3D& va = weldedVertices[static_cast<int>(a)];
        const QVector3D& vb = weldedVertices[static_cast<int>(b)];
        const QVector3D& vc = weldedVertices[static_cast<int>(c)];
        if (QVector3D::crossProduct(vb - va, vc - va).lengthSquared() <= 1e-20f) {
            ++stats.removedDegenerateFaces;
            continue;
        }

        const FaceKey faceKey = makeFaceKey(a, b, c);
        if (faceMap.find(faceKey) != faceMap.end()) {
            ++stats.removedDuplicateFaces;
            continue;
        }

        faceMap.emplace(faceKey, faces.size());
        faces.append(TriangleIndices{a, b, c});
    }

    if (faces.isEmpty()) {
        cleanedMesh.vertices = weldedVertices;
        if (statsOut) {
            stats.weldedVertexCount = cleanedMesh.vertices.size();
            *statsOut = stats;
        }
        return cleanedMesh;
    }

    const int faceCount = faces.size();
    const int componentThreshold = qMax(32, faceCount / 50000);
    stats.componentFaceThreshold = componentThreshold;

    DisjointSet disjointSet(faceCount);
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edgeOwner;
    edgeOwner.reserve(static_cast<std::size_t>(faceCount) * 2);

    for (int faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        const TriangleIndices& face = faces[faceIndex];
        const EdgeKey edges[3] = {
            makeEdgeKey(face.a, face.b),
            makeEdgeKey(face.b, face.c),
            makeEdgeKey(face.c, face.a)
        };

        for (const EdgeKey& edge : edges) {
            const auto it = edgeOwner.find(edge);
            if (it == edgeOwner.end()) {
                edgeOwner.emplace(edge, faceIndex);
            } else {
                disjointSet.unite(faceIndex, it->second);
            }
        }
    }

    QVector<int> rootByFace(faceCount, -1);
    std::unordered_map<int, int> componentSizes;
    componentSizes.reserve(static_cast<std::size_t>(faceCount));
    int largestRoot = -1;
    int largestSize = 0;

    for (int faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        const int root = disjointSet.find(faceIndex);
        rootByFace[faceIndex] = root;
        const int newSize = ++componentSizes[root];
        if (newSize > largestSize) {
            largestSize = newSize;
            largestRoot = root;
        }
    }

    stats.largestComponentRatio = (faceCount > 0)
        ? static_cast<float>(largestSize) / static_cast<float>(faceCount)
        : 1.0f;

    std::unordered_map<int, bool> keepRoot;
    keepRoot.reserve(componentSizes.size());
    int keptFaceCount = 0;
    for (const auto& entry : componentSizes) {
        if (entry.second >= componentThreshold || entry.first == largestRoot) {
            keepRoot.emplace(entry.first, true);
            ++stats.keptComponents;
            keptFaceCount += entry.second;
        }
    }

    // If the connectivity graph is highly fragmented, component pruning can destroy
    // valid anatomy. In that case, keep all components and only rely on other cleanup.
    const int minFacesToKeep = static_cast<int>(std::lround(faceCount * 0.85));
    if (keptFaceCount < minFacesToKeep) {
        keepRoot.clear();
        keepRoot.reserve(componentSizes.size());
        for (const auto& entry : componentSizes) {
            keepRoot.emplace(entry.first, true);
        }
        stats.keptComponents = static_cast<int>(componentSizes.size());
        stats.componentFilteringApplied = false;
    } else {
        stats.componentFilteringApplied = true;
    }

    QVector<int> finalVertexRemap(weldedVertices.size(), -1);
    QVector<QVector3D> finalVertices;
    finalVertices.reserve(weldedVertices.size());
    QVector<unsigned int> finalIndices;
    finalIndices.reserve(faces.size() * 3);

    for (int faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        if (keepRoot.find(rootByFace[faceIndex]) == keepRoot.end()) {
            ++stats.removedSmallComponentFaces;
            continue;
        }

        const TriangleIndices& face = faces[faceIndex];
        const unsigned int faceVertices[3] = {face.a, face.b, face.c};
        unsigned int remapped[3] = {0, 0, 0};

        bool valid = true;
        for (int i = 0; i < 3; ++i) {
            const unsigned int sourceIndex = faceVertices[i];
            if (sourceIndex >= static_cast<unsigned int>(finalVertexRemap.size())) {
                valid = false;
                break;
            }

            int mapped = finalVertexRemap[static_cast<int>(sourceIndex)];
            if (mapped < 0) {
                mapped = finalVertices.size();
                finalVertexRemap[static_cast<int>(sourceIndex)] = mapped;
                finalVertices.append(weldedVertices[static_cast<int>(sourceIndex)]);
            }
            remapped[i] = static_cast<unsigned int>(mapped);
        }

        if (!valid) {
            continue;
        }

        if (remapped[0] == remapped[1] || remapped[1] == remapped[2] || remapped[0] == remapped[2]) {
            continue;
        }

        finalIndices.append(remapped[0]);
        finalIndices.append(remapped[1]);
        finalIndices.append(remapped[2]);
    }

    cleanedMesh.vertices = finalVertices;
    cleanedMesh.indices = finalIndices;
    stats.weldedVertexCount = cleanedMesh.vertices.size();

    if (statsOut) {
        *statsOut = stats;
    }
    return cleanedMesh;
}

// ---------------------------------------------------------------------------
// Removes faces whose edges are already shared by 2 faces (non-manifold edges).
// OpenMesh refuses these with "complex edge" errors, so strip them first.
// ---------------------------------------------------------------------------
MarchingCubes::Mesh filterNonManifoldFaces(const MarchingCubes::Mesh& mesh)
{
    const int faceCount = mesh.indices.size() / 3;
    if (faceCount == 0) return mesh;

    std::unordered_map<EdgeKey, int, EdgeKeyHash> edgeUse;
    edgeUse.reserve(static_cast<std::size_t>(faceCount) * 2);

    QVector<bool> accepted(faceCount, false);
    int removedCount = 0;

    for (int i = 0; i < faceCount; ++i) {
        const unsigned int a = mesh.indices[i * 3 + 0];
        const unsigned int b = mesh.indices[i * 3 + 1];
        const unsigned int c = mesh.indices[i * 3 + 2];
        const EdgeKey e0 = makeEdgeKey(a, b);
        const EdgeKey e1 = makeEdgeKey(b, c);
        const EdgeKey e2 = makeEdgeKey(c, a);
        if (edgeUse[e0] < 2 && edgeUse[e1] < 2 && edgeUse[e2] < 2) {
            accepted[i] = true;
            ++edgeUse[e0]; ++edgeUse[e1]; ++edgeUse[e2];
        } else {
            ++removedCount;
        }
    }

    if (removedCount == 0) return mesh;

    QVector<bool> vertexUsed(mesh.vertices.size(), false);
    QVector<unsigned int> newIndices;
    newIndices.reserve(static_cast<std::size_t>(faceCount - removedCount) * 3);

    for (int i = 0; i < faceCount; ++i) {
        if (!accepted[i]) continue;
        const unsigned int a = mesh.indices[i * 3 + 0];
        const unsigned int b = mesh.indices[i * 3 + 1];
        const unsigned int c = mesh.indices[i * 3 + 2];
        vertexUsed[a] = vertexUsed[b] = vertexUsed[c] = true;
        newIndices.append(a); newIndices.append(b); newIndices.append(c);
    }

    QVector<int> vRemap(mesh.vertices.size(), -1);
    QVector<QVector3D> newVertices;
    newVertices.reserve(mesh.vertices.size());
    for (int i = 0; i < mesh.vertices.size(); ++i) {
        if (vertexUsed[i]) { vRemap[i] = newVertices.size(); newVertices.append(mesh.vertices[i]); }
    }
    for (unsigned int& idx : newIndices)
        idx = static_cast<unsigned int>(vRemap[static_cast<int>(idx)]);

    qDebug() << "[Simplifier] filterNonManifoldFaces: removed" << removedCount
             << "of" << faceCount << "faces";
    MarchingCubes::Mesh result;
    result.vertices = newVertices;
    result.indices  = newIndices;
    return result;
}

// ---------------------------------------------------------------------------
// Fast O(N) vertex-clustering simplification.
// Groups vertices into a regular grid and merges each cell to its centroid.
// Used as a pre-pass before OpenMesh QEM when the mesh is very large.
// ---------------------------------------------------------------------------
MarchingCubes::Mesh vertexClusterSimplify(const MarchingCubes::Mesh& mesh, int targetFaceCount)
{
    const int faceCount = mesh.indices.size() / 3;
    if (faceCount <= targetFaceCount || mesh.vertices.isEmpty()) return mesh;

    QVector3D bbMin = mesh.vertices.first(), bbMax = mesh.vertices.first();
    for (const QVector3D& v : mesh.vertices) {
        bbMin.setX(qMin(bbMin.x(), v.x())); bbMin.setY(qMin(bbMin.y(), v.y())); bbMin.setZ(qMin(bbMin.z(), v.z()));
        bbMax.setX(qMax(bbMax.x(), v.x())); bbMax.setY(qMax(bbMax.y(), v.y())); bbMax.setZ(qMax(bbMax.z(), v.z()));
    }
    QVector3D bbSize = bbMax - bbMin;
    const float eps = 1e-6f;
    if (bbSize.x() < eps) bbSize.setX(eps);
    if (bbSize.y() < eps) bbSize.setY(eps);
    if (bbSize.z() < eps) bbSize.setZ(eps);

    // grid resolution: for a surface, output faces ≈ 2 * g²;
    // so g ≈ sqrt(targetFaceCount / 2) gives a reasonable cell size.
    const int g = qMax(8, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(targetFaceCount) / 2.0))));

    struct CellAccum { QVector3D sum; int count = 0; };
    std::unordered_map<std::int64_t, CellAccum> cellMap;
    cellMap.reserve(static_cast<std::size_t>(mesh.vertices.size()));

    auto cellIdx = [&](const QVector3D& v) -> std::int64_t {
        const int cx = qBound(0, static_cast<int>(((v.x() - bbMin.x()) / bbSize.x()) * g), g - 1);
        const int cy = qBound(0, static_cast<int>(((v.y() - bbMin.y()) / bbSize.y()) * g), g - 1);
        const int cz = qBound(0, static_cast<int>(((v.z() - bbMin.z()) / bbSize.z()) * g), g - 1);
        return static_cast<std::int64_t>(cx)
             + static_cast<std::int64_t>(cy) * g
             + static_cast<std::int64_t>(cz) * g * g;
    };

    for (const QVector3D& v : mesh.vertices) {
        auto& acc = cellMap[cellIdx(v)];
        acc.sum += v; ++acc.count;
    }

    std::unordered_map<std::int64_t, unsigned int> cellToVtx;
    cellToVtx.reserve(cellMap.size());
    QVector<QVector3D> newVerts;
    newVerts.reserve(static_cast<int>(cellMap.size()));
    for (auto& [cell, acc] : cellMap) {
        cellToVtx[cell] = static_cast<unsigned int>(newVerts.size());
        newVerts.append(acc.sum / static_cast<float>(acc.count));
    }

    QVector<unsigned int> vtxToNew(mesh.vertices.size());
    for (int i = 0; i < mesh.vertices.size(); ++i)
        vtxToNew[i] = cellToVtx[cellIdx(mesh.vertices[i])];

    std::unordered_map<FaceKey, bool, FaceKeyHash> seen;
    seen.reserve(static_cast<std::size_t>(faceCount));
    QVector<unsigned int> newIdx;
    newIdx.reserve(mesh.indices.size());

    for (int i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const unsigned int a = vtxToNew[mesh.indices[i + 0]];
        const unsigned int b = vtxToNew[mesh.indices[i + 1]];
        const unsigned int c = vtxToNew[mesh.indices[i + 2]];
        if (a == b || b == c || a == c) continue;
        if (!seen.emplace(makeFaceKey(a, b, c), true).second) continue;
        newIdx.append(a); newIdx.append(b); newIdx.append(c);
    }

    qDebug() << "[Simplifier] vertexClusterSimplify (g=" << g << "):" << faceCount
             << "->" << (newIdx.size() / 3) << "faces";
    MarchingCubes::Mesh result;
    result.vertices = newVerts;
    result.indices  = newIdx;
    return result;
}

// ---------------------------------------------------------------------------
// Memory-efficient pre-cluster for very large meshes (>500k faces).
// Identical algorithm to vertexClusterSimplify but:
//   (1) cellMap reserved to min(N, g³) instead of N — avoids a huge bucket
//       array when N >> number of occupied grid cells on a surface mesh.
//   (2) The per-face deduplication map ("seen", ~1.5 GB for 30M faces) is
//       omitted entirely.  Duplicate clustered faces are removed cheaply by
//       the subsequent cleanupMeshTopology pass that runs on the small output.
// Combined effect: peak memory drops from ~4 GB to ~300 MB on a 30M-face mesh,
// cutting this pass from 3+ minutes to ~30 seconds.
// ---------------------------------------------------------------------------
MarchingCubes::Mesh fastPreCluster(const MarchingCubes::Mesh& mesh, int targetFaceCount)
{
    const int faceCount = mesh.indices.size() / 3;
    if (faceCount <= targetFaceCount || mesh.vertices.isEmpty()) return mesh;

    QVector3D bbMin = mesh.vertices.first(), bbMax = mesh.vertices.first();
    for (const QVector3D& v : mesh.vertices) {
        bbMin.setX(qMin(bbMin.x(), v.x())); bbMin.setY(qMin(bbMin.y(), v.y())); bbMin.setZ(qMin(bbMin.z(), v.z()));
        bbMax.setX(qMax(bbMax.x(), v.x())); bbMax.setY(qMax(bbMax.y(), v.y())); bbMax.setZ(qMax(bbMax.z(), v.z()));
    }
    QVector3D bbSize = bbMax - bbMin;
    const float eps = 1e-6f;
    if (bbSize.x() < eps) bbSize.setX(eps);
    if (bbSize.y() < eps) bbSize.setY(eps);
    if (bbSize.z() < eps) bbSize.setZ(eps);

    const int g = qMax(8, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(targetFaceCount) / 2.0))));

    struct CellAccum { QVector3D sum; int count = 0; };
    std::unordered_map<std::int64_t, CellAccum> cellMap;
    // Reserve at most g³ buckets — for g≈274 (150k target) that is ~20 M slots
    // (160 MB) instead of vertex_count*8 B = 240 MB, and crucially the "seen"
    // map below (which would be ~1.5 GB) is never allocated at all.
    const std::size_t maxCells = static_cast<std::size_t>(g)
                               * static_cast<std::size_t>(g)
                               * static_cast<std::size_t>(g);
    cellMap.reserve(qMin(static_cast<std::size_t>(mesh.vertices.size()), maxCells));

    auto cellIdx = [&](const QVector3D& v) -> std::int64_t {
        const int cx = qBound(0, static_cast<int>(((v.x() - bbMin.x()) / bbSize.x()) * g), g - 1);
        const int cy = qBound(0, static_cast<int>(((v.y() - bbMin.y()) / bbSize.y()) * g), g - 1);
        const int cz = qBound(0, static_cast<int>(((v.z() - bbMin.z()) / bbSize.z()) * g), g - 1);
        return static_cast<std::int64_t>(cx)
             + static_cast<std::int64_t>(cy) * g
             + static_cast<std::int64_t>(cz) * g * g;
    };

    for (const QVector3D& v : mesh.vertices) {
        auto& acc = cellMap[cellIdx(v)];
        acc.sum += v; ++acc.count;
    }

    std::unordered_map<std::int64_t, unsigned int> cellToVtx;
    cellToVtx.reserve(cellMap.size());
    QVector<QVector3D> newVerts;
    newVerts.reserve(static_cast<int>(cellMap.size()));
    for (auto& [cell, acc] : cellMap) {
        cellToVtx[cell] = static_cast<unsigned int>(newVerts.size());
        newVerts.append(acc.sum / static_cast<float>(acc.count));
    }

    QVector<unsigned int> vtxToNew(mesh.vertices.size());
    for (int i = 0; i < mesh.vertices.size(); ++i)
        vtxToNew[i] = cellToVtx[cellIdx(mesh.vertices[i])];

    // No face-dedup map.  For a surface mesh output faces ≈ 2*g² ≈ 2*newVerts
    // so reserve ~6 index slots per output vertex as a tight upper bound.
    QVector<unsigned int> newIdx;
    newIdx.reserve(static_cast<int>(newVerts.size()) * 6);
    for (int i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const unsigned int a = vtxToNew[mesh.indices[i + 0]];
        const unsigned int b = vtxToNew[mesh.indices[i + 1]];
        const unsigned int c = vtxToNew[mesh.indices[i + 2]];
        if (a == b || b == c || a == c) continue;
        newIdx.append(a); newIdx.append(b); newIdx.append(c);
    }

    qDebug() << "[Simplifier] fastPreCluster (g=" << g << "):" << faceCount
             << "->" << (newIdx.size() / 3) << "faces (no face-dedup; cleanup handles it)";
    MarchingCubes::Mesh result;
    result.vertices = newVerts;
    result.indices  = newIdx;
    return result;
}

int clampTargetFaceCount(int targetFaceCount, int inputFaceCount, double aggressiveness)
{
    if (inputFaceCount <= 0) {
        return 0;
    }

    const int safeTarget = std::clamp(targetFaceCount, 1, inputFaceCount);
    const double safeAggressiveness = std::clamp(aggressiveness, 0.25, 8.0);
    const int adjustedTarget = std::clamp(int(std::lround(double(safeTarget) / safeAggressiveness)), 1, inputFaceCount);
    return adjustedTarget;
}

#if MESH_SIMPLIFIER_HAS_OPENMESH
OpenMeshType toOpenMesh(const MarchingCubes::Mesh& inputMesh)
{
    OpenMeshType mesh;
    mesh.request_vertex_status();
    mesh.request_edge_status();
    mesh.request_face_status();

    QVector<OpenMeshType::VertexHandle> handles;
    handles.reserve(inputMesh.vertices.size());

    for (const QVector3D& vertex : inputMesh.vertices) {
        handles.append(mesh.add_vertex(OpenMeshType::Point(vertex.x(), vertex.y(), vertex.z())));
    }

    for (int i = 0; i + 2 < inputMesh.indices.size(); i += 3) {
        const unsigned int i0 = inputMesh.indices[i + 0];
        const unsigned int i1 = inputMesh.indices[i + 1];
        const unsigned int i2 = inputMesh.indices[i + 2];
        if (i0 >= static_cast<unsigned int>(handles.size()) ||
            i1 >= static_cast<unsigned int>(handles.size()) ||
            i2 >= static_cast<unsigned int>(handles.size())) {
            continue;
        }

        const QVector3D v0 = inputMesh.vertices[static_cast<int>(i0)];
        const QVector3D v1 = inputMesh.vertices[static_cast<int>(i1)];
        const QVector3D v2 = inputMesh.vertices[static_cast<int>(i2)];
        if (QVector3D::crossProduct(v1 - v0, v2 - v0).lengthSquared() <= 1e-20f) {
            continue;
        }

        const OpenMeshType::FaceHandle faceHandle = mesh.add_face(handles[static_cast<int>(i0)],
                                                                  handles[static_cast<int>(i1)],
                                                                  handles[static_cast<int>(i2)]);
        if (!faceHandle.is_valid()) {
            continue;
        }
    }

    return mesh;
}

MarchingCubes::Mesh fromOpenMesh(OpenMeshType& mesh)
{
    MarchingCubes::Mesh output;

    mesh.garbage_collection();

    QVector<int> vertexRemap(mesh.n_vertices(), -1);
    QVector<OpenMeshType::VertexHandle> usedVertices;
    usedVertices.reserve(mesh.n_vertices());

    for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end(); ++faceIt) {
        for (auto fvIt = mesh.fv_iter(*faceIt); fvIt.is_valid(); ++fvIt) {
            const int index = fvIt->idx();
            if (index >= 0 && index < vertexRemap.size() && vertexRemap[index] < 0) {
                vertexRemap[index] = -2;
                usedVertices.append(*fvIt);
            }
        }
    }

    for (const OpenMeshType::VertexHandle& handle : usedVertices) {
        const int idx = handle.idx();
        if (idx < 0 || idx >= vertexRemap.size()) {
            continue;
        }
        vertexRemap[idx] = output.vertices.size();
        const OpenMeshType::Point point = mesh.point(handle);
        output.vertices.append(QVector3D(point[0], point[1], point[2]));
    }

    for (auto faceIt = mesh.faces_begin(); faceIt != mesh.faces_end(); ++faceIt) {
        QVector<unsigned int> faceIndices;
        faceIndices.reserve(3);
        for (auto fvIt = mesh.fv_iter(*faceIt); fvIt.is_valid(); ++fvIt) {
            const int remapped = (fvIt->idx() >= 0 && fvIt->idx() < vertexRemap.size()) ? vertexRemap[fvIt->idx()] : -1;
            if (remapped < 0) {
                faceIndices.clear();
                break;
            }
            faceIndices.append(static_cast<unsigned int>(remapped));
        }

        if (faceIndices.size() == 3) {
            const QVector3D& a = output.vertices[static_cast<int>(faceIndices[0])];
            const QVector3D& b = output.vertices[static_cast<int>(faceIndices[1])];
            const QVector3D& c = output.vertices[static_cast<int>(faceIndices[2])];
            if (QVector3D::crossProduct(b - a, c - a).lengthSquared() > 1e-20f) {
                output.indices.append(faceIndices[0]);
                output.indices.append(faceIndices[1]);
                output.indices.append(faceIndices[2]);
            }
        }
    }

    return output;
}
#endif

} // namespace

bool isOpenMeshAvailable()
{
    return MESH_SIMPLIFIER_HAS_OPENMESH != 0;
}

QString backendName()
{
    return isOpenMeshAvailable() ? QStringLiteral("OpenMesh QEM") : QStringLiteral("OpenMesh unavailable");
}

SimplifyReport simplifyMeshDetailed(const MarchingCubes::Mesh& inputMesh, int targetFaceCount, double aggressiveness)
{
    SimplifyReport report;
    report.inputFaceCount = inputMesh.indices.size() / 3;

    qDebug().noquote() << QString("[Simplifier] request inputVertices=%1 inputFaces=%2 targetFaces=%3 aggressiveness=%4 backend=%5")
                              .arg(inputMesh.vertices.size())
                              .arg(report.inputFaceCount)
                              .arg(targetFaceCount)
                              .arg(aggressiveness, 0, 'f', 2)
                              .arg(backendName());

    // For very large meshes, pre-cluster FIRST before any topology cleanup.
    // cleanupMeshTopology builds an edge-adjacency map with 3*F entries; on a
    // 30M-face mesh that is 90M map entries (~4-6 GB peak), causing OS thrashing
    // and 10+ min runtimes.  fastPreCluster skips the face-dedup map and caps
    // the bucket reserve at g³ so it completes in ~30 s, then cleanup runs on
    // the much smaller output (~150k faces) and takes < 1 s.
    static constexpr int kPreClusterThreshold = 500'000;
    static constexpr int kPreClusterTarget    = 150'000;

    MarchingCubes::Mesh preClusteredStorage;
    const MarchingCubes::Mesh* cleanupInputPtr = &inputMesh;
    if (report.inputFaceCount > kPreClusterThreshold) {
        preClusteredStorage = fastPreCluster(inputMesh, kPreClusterTarget);
        cleanupInputPtr = &preClusteredStorage;
        qDebug() << "[Simplifier] fastPreCluster output:" << (preClusteredStorage.indices.size() / 3) << "faces.";
    }
    const MarchingCubes::Mesh& cleanupInput = *cleanupInputPtr;

    CleanupStats preCleanupStats;
    MarchingCubes::Mesh workingMesh = cleanupMeshTopology(cleanupInput, &preCleanupStats);
    const int workingFaceCount = workingMesh.indices.size() / 3;

    qDebug() << "[Simplifier] cleanup pre-pass: vertices" << cleanupInput.vertices.size() << "->" << workingMesh.vertices.size()
             << ", faces" << (cleanupInput.indices.size() / 3) << "->" << workingFaceCount
             << ", duplicate vertices removed" << preCleanupStats.removedDuplicateVertices
             << ", degenerate faces removed" << preCleanupStats.removedDegenerateFaces
             << ", duplicate faces removed" << preCleanupStats.removedDuplicateFaces
             << ", tiny component faces removed" << preCleanupStats.removedSmallComponentFaces
             << ", component threshold" << preCleanupStats.componentFaceThreshold
             << ", component filtering applied" << preCleanupStats.componentFilteringApplied
             << ", largest component ratio" << preCleanupStats.largestComponentRatio;

    if (workingMesh.vertices.isEmpty() || workingMesh.indices.size() < 3) {
        report.success = true;
        report.mesh = workingMesh;
        report.outputFaceCount = workingFaceCount;
        report.message = QStringLiteral("Mesh is empty after topology cleanup; nothing to simplify.");
        return report;
    }

    // Feed the cleaned mesh directly into QEM — no O(N²) pre-pass needed.
    MarchingCubes::Mesh simplificationInput = workingMesh;

    // Pre-pass 1: Remove non-manifold edges so OpenMesh receives a clean mesh
    // and stops emitting "complex edge" warnings for every bad triangle.
    simplificationInput = filterNonManifoldFaces(simplificationInput);

    // Safety cluster pass: with the fastPreCluster pre-pass the mesh should
    // already be ≤150k faces here, but act as a safety net for edge cases.
    static constexpr int kQemFaceLimit = 100'000;
    if (simplificationInput.indices.size() / 3 > kQemFaceLimit) {
        const int clusterTarget = qMin(kQemFaceLimit, qMax(targetFaceCount, 50'000));
        qDebug() << "[Simplifier] Safety cluster (" << (simplificationInput.indices.size() / 3)
                 << " faces); vertex-clustering to ~" << clusterTarget << " faces.";
        simplificationInput = vertexClusterSimplify(simplificationInput, clusterTarget);
        qDebug() << "[Simplifier] After safety cluster:" << (simplificationInput.indices.size() / 3) << "faces.";
    }

    const int effectiveTarget = clampTargetFaceCount(targetFaceCount, simplificationInput.indices.size() / 3, aggressiveness);
    qDebug() << "[Simplifier] effective target faces:" << effectiveTarget;
    if (effectiveTarget >= simplificationInput.indices.size() / 3) {
        report.success = true;
        report.mesh = simplificationInput;
        report.outputFaceCount = simplificationInput.indices.size() / 3;
        report.message = QStringLiteral("Target face count is not lower than the cleaned mesh face count.");
        qDebug() << "[Simplifier] skipping simplification because target is not lower than input.";
        return report;
    }

#if !MESH_SIMPLIFIER_HAS_OPENMESH
    report.mesh = simplificationInput;
    report.outputFaceCount = simplificationInput.indices.size() / 3;
    report.message = QStringLiteral("OpenMesh is not available in this build; simplification skipped.");
    report.success = false;
    qWarning() << report.message;
    return report;
#else
    // OpenMesh prints "PolyMeshT::add_face: complex edge/vertex" to std::cerr for
    // every non-manifold triangle it rejects.  On large scans this produces thousands
    // of lines.  Silence it by redirecting cerr to a null sink for this call only.
    OpenMeshType mesh = [&simplificationInput]() {
        struct NullBuf : std::streambuf {
            int overflow(int c) override { return c; }
        } nullBuf;
        std::streambuf* const oldBuf = std::cerr.rdbuf(&nullBuf);
        OpenMeshType m = toOpenMesh(simplificationInput);
        std::cerr.rdbuf(oldBuf);
        return m;
    }();
    qDebug() << "[Simplifier] OpenMesh conversion produced faces:" << mesh.n_faces()
             << "vertices:" << mesh.n_vertices();
    if (mesh.n_faces() == 0) {
        report.mesh = workingMesh;
        report.outputFaceCount = workingFaceCount;
        report.message = QStringLiteral("OpenMesh conversion produced no valid faces.");
        report.success = false;
        qWarning() << report.message;
        return report;
    }

    using DecimaterType = OpenMesh::Decimater::DecimaterT<OpenMeshType>;
    using QuadricModule = OpenMesh::Decimater::ModQuadricT<OpenMeshType>;

    DecimaterType decimater(mesh);
    QuadricModule::Handle quadricHandle;
    decimater.add(quadricHandle);
    if (!decimater.initialize()) {
        report.mesh = simplificationInput;
        report.outputFaceCount = simplificationInput.indices.size() / 3;
        report.message = QStringLiteral("OpenMesh decimater initialization failed.");
        report.success = false;
        qWarning() << report.message;
        return report;
    }

    qDebug() << "[Simplifier] decimater initialized; starting decimation.";

    qDebug().noquote() << QString(
        "=== QEM Decimation START ===\n"
        "  Input faces   : %1\n"
        "  Target faces  : %2\n"
        "  Aggressiveness: %3\n"
        "  Backend       : %4"
    ).arg(simplificationInput.indices.size() / 3)
     .arg(effectiveTarget)
     .arg(aggressiveness, 0, 'f', 2)
     .arg(backendName());

    QElapsedTimer qemTimer;
    qemTimer.start();
    const bool decimationOk = decimater.decimate_to_faces(static_cast<unsigned int>(effectiveTarget));
    const qint64 qemElapsedMs = qemTimer.elapsed();
    qDebug() << "[Simplifier] QEM decimation finished ok=" << decimationOk
             << "elapsed:" << qemElapsedMs << "ms";
    mesh.garbage_collection();
    qDebug() << "[Simplifier] post-GC faces:" << mesh.n_faces()
             << "vertices:" << mesh.n_vertices();

    // Check if we're close to the target or if decimation failed due to topological issues
    const unsigned int actualFaces = mesh.n_faces();
    const double reductionRatio = (effectiveTarget > 0) ? double(actualFaces) / double(effectiveTarget) : 1.0;
    
    if (!decimationOk && actualFaces > static_cast<unsigned int>(effectiveTarget)) {
        // Decimation couldn't complete fully, but we got some reduction
        // This often happens with meshes containing "complex edges" (topological issues)
        qWarning() << QString("[Simplifier] decimation incomplete: target=%1 actual=%2 ratio=%3x (likely due to mesh topology issues)")
                      .arg(effectiveTarget).arg(actualFaces).arg(reductionRatio, 0, 'f', 2);
    } else if (!decimationOk) {
        report.mesh = simplificationInput;
        report.outputFaceCount = simplificationInput.indices.size() / 3;
        report.message = QStringLiteral("OpenMesh decimation did not complete successfully.");
        report.success = false;
        qWarning() << report.message;
        return report;
    }

    CleanupStats postCleanupStats;
    report.mesh = cleanupMeshTopology(fromOpenMesh(mesh), &postCleanupStats);
    report.outputFaceCount = report.mesh.indices.size() / 3;
    report.success = !report.mesh.vertices.isEmpty() && !report.mesh.indices.isEmpty();
    report.message = report.success
        ? QStringLiteral("Mesh simplification completed successfully.")
        : QStringLiteral("Mesh simplification returned an empty mesh.");

    qDebug() << "[Simplifier] cleanup post-pass: vertices" << mesh.n_vertices() << "->" << report.mesh.vertices.size()
             << ", faces" << mesh.n_faces() << "->" << report.outputFaceCount
             << ", duplicate vertices removed" << postCleanupStats.removedDuplicateVertices
             << ", degenerate faces removed" << postCleanupStats.removedDegenerateFaces
             << ", duplicate faces removed" << postCleanupStats.removedDuplicateFaces
             << ", tiny component faces removed" << postCleanupStats.removedSmallComponentFaces
             << ", component threshold" << postCleanupStats.componentFaceThreshold
             << ", component filtering applied" << postCleanupStats.componentFilteringApplied
             << ", largest component ratio" << postCleanupStats.largestComponentRatio;

    if (report.outputFaceCount == report.inputFaceCount) {
        qWarning() << "[Simplifier] output face count matches input face count; decimation may have been unable to collapse any edges.";
    }

    const double reductionPct = (report.inputFaceCount > 0)
        ? (1.0 - double(report.outputFaceCount) / double(report.inputFaceCount)) * 100.0
        : 0.0;
    qDebug().noquote() << QString(
        "=== QEM Decimation COMPLETE ===\n"
        "  Input faces : %1\n"
        "  Target faces: %2\n"
        "  Output faces: %3\n"
        "  Reduction   : %4%\n"
        "  QEM time    : %5 ms\n"
        "  Success     : %6"
    ).arg(report.inputFaceCount)
     .arg(effectiveTarget)
     .arg(report.outputFaceCount)
     .arg(reductionPct, 0, 'f', 1)
     .arg(qemElapsedMs)
     .arg(report.success ? "YES" : "NO");
    return report;
#endif
}

MarchingCubes::Mesh simplifyMesh(const MarchingCubes::Mesh& inputMesh, int targetFaceCount, double aggressiveness)
{
    return simplifyMeshDetailed(inputMesh, targetFaceCount, aggressiveness).mesh;
}

} // namespace MeshSimplifier
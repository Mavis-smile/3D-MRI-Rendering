#include "MeshSimplifier.h"

#include <QDebug>
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
/*
analyze a 3D mesh and determine which triangles are:
- Outside surface triangles
- Internal cavity boundary triangles
- Completely internal triangles */
// Triangle classification for medical anatomy preservation
enum class TriangleType {
    External = 0,           // External surface - can be simplified
    CavityBoundary = 1,     // Bounds hollow spaces (medically important) - keep 100%
    FullyInternal = 2       // Completely enclosed - can be removed
};

// Classify triangles by their topological position using winding number
// Returns a vector with classification for each triangle
QVector<TriangleType> classifyTrianglesByWindingNumber(const MarchingCubes::Mesh& mesh) {
    const int triangleCount = mesh.indices.size() / 3;
    QVector<TriangleType> classification(triangleCount, TriangleType::External);
    
    if (triangleCount == 0) {
        return classification;
    }
    
    // Compute AABB for ray-casting normalization
    QVector3D aabbMin = mesh.vertices.first();
    QVector3D aabbMax = mesh.vertices.first();
    
    for (const QVector3D& v : mesh.vertices) {
        aabbMin.setX(qMin(aabbMin.x(), v.x()));
        aabbMin.setY(qMin(aabbMin.y(), v.y()));
        aabbMin.setZ(qMin(aabbMin.z(), v.z()));
        aabbMax.setX(qMax(aabbMax.x(), v.x()));
        aabbMax.setY(qMax(aabbMax.y(), v.y()));
        aabbMax.setZ(qMax(aabbMax.z(), v.z()));
    }
    
    const QVector3D aabbSize = aabbMax - aabbMin;
    const float rayLength = aabbSize.length() * 2.0f;
    
    // For each triangle, cast a ray from its centroid to compute winding number
    for (int triIdx = 0; triIdx < triangleCount; ++triIdx) {
        const int idx0 = mesh.indices[triIdx * 3 + 0];
        const int idx1 = mesh.indices[triIdx * 3 + 1];
        const int idx2 = mesh.indices[triIdx * 3 + 2];
        
        if (idx0 >= mesh.vertices.size() || idx1 >= mesh.vertices.size() || idx2 >= mesh.vertices.size()) {
            continue;
        }
        
        const QVector3D& v0 = mesh.vertices[idx0];
        const QVector3D& v1 = mesh.vertices[idx1];
        const QVector3D& v2 = mesh.vertices[idx2];
        
        // Triangle centroid
        const QVector3D centroid = (v0 + v1 + v2) / 3.0f;
        
        // Ray direction (arbitrary but consistent)
        const QVector3D rayDir = QVector3D(1.0f, 0.3f, 0.7f).normalized();
        const QVector3D rayEnd = centroid + rayDir * rayLength;
        
        // Compute winding number by counting ray-triangle intersections
        int windingNumber = 0;
        for (int i = 0; i < triangleCount; ++i) {
            if (i == triIdx) continue; // Skip self
            
            const int ti0 = mesh.indices[i * 3 + 0];
            const int ti1 = mesh.indices[i * 3 + 1];
            const int ti2 = mesh.indices[i * 3 + 2];
            
            if (ti0 >= mesh.vertices.size() || ti1 >= mesh.vertices.size() || ti2 >= mesh.vertices.size()) {
                continue;
            }
            
            const QVector3D& tv0 = mesh.vertices[ti0];
            const QVector3D& tv1 = mesh.vertices[ti1];
            const QVector3D& tv2 = mesh.vertices[ti2];
            
            // Ray-triangle intersection using Möller-Trumbore algorithm
            const float epsilon = 1e-8f;
            const QVector3D edge1 = tv1 - tv0;
            const QVector3D edge2 = tv2 - tv0;
            const QVector3D rayVec = rayEnd - centroid;
            
            const QVector3D h = QVector3D::crossProduct(rayVec, edge2);
            const float det = QVector3D::dotProduct(edge1, h);
            
            if (std::abs(det) < epsilon) {
                continue; // Ray is parallel to triangle
            }
            
            const float invDet = 1.0f / det;
            const QVector3D s = centroid - tv0;
            const float u = invDet * QVector3D::dotProduct(s, h);
            
            if (u < 0.0f || u > 1.0f) {
                continue;
            }
            
            const QVector3D q = QVector3D::crossProduct(s, edge1);
            const float v = invDet * QVector3D::dotProduct(rayVec, q);
            
            if (v < 0.0f || u + v > 1.0f) {
                continue;
            }
            
            const float t = invDet * QVector3D::dotProduct(edge2, q);
            
            if (t > epsilon && t < 1.0f - epsilon) {
                // Ray hits triangle - check winding contribution via normal orientation
                const QVector3D normal = QVector3D::crossProduct(edge1, edge2);
                if (QVector3D::dotProduct(normal, rayVec) > 0.0f) {
                    ++windingNumber;
                } else {
                    --windingNumber;
                }
            }
        }
        
        // Classify based on winding number
        // External: winding number = 0 (outside)
        // CavityBoundary: winding number = ±1 (on boundary, typically manifold)
        // FullyInternal: |winding number| >= 2 (deeply enclosed)
        const int absWinding = std::abs(windingNumber);
        if (absWinding == 0) {
            classification[triIdx] = TriangleType::External;
        } else if (absWinding == 1) {
            classification[triIdx] = TriangleType::CavityBoundary;
        } else {
            classification[triIdx] = TriangleType::FullyInternal;
        }
    }
    
    // Log classification summary
    int extCount = 0, cavityCount = 0, internalCount = 0;
    for (TriangleType type : classification) {
        if (type == TriangleType::External) ++extCount;
        else if (type == TriangleType::CavityBoundary) ++cavityCount;
        else ++internalCount;
    }
    
    qDebug() << "[Simplifier] Triangle classification: External=" << extCount 
             << "(" << (100.0 * extCount / triangleCount) << "%) Cavity=" << cavityCount
             << "(" << (100.0 * cavityCount / triangleCount) << "%) Internal=" << internalCount
             << "(" << (100.0 * internalCount / triangleCount) << "%)";
    
    return classification;
}

// Remove fully internal triangles and return simplified mesh
MarchingCubes::Mesh removeFullyInternalTriangles(const MarchingCubes::Mesh& mesh, const QVector<TriangleType>& classification) {
    MarchingCubes::Mesh result;
    
    const int triangleCount = mesh.indices.size() / 3;
    if (triangleCount == 0 || classification.size() != triangleCount) {
        return mesh;
    }
    
    // Mark which vertices are used by non-internal triangles
    QVector<bool> vertexUsed(mesh.vertices.size(), false);
    QVector<unsigned int> indices;
    indices.reserve(mesh.indices.size());
    
    int removedCount = 0;
    for (int i = 0; i < triangleCount; ++i) {
        if (classification[i] == TriangleType::FullyInternal) {
            ++removedCount;
            continue;
        }
        
        const int idx0 = mesh.indices[i * 3 + 0];
        const int idx1 = mesh.indices[i * 3 + 1];
        const int idx2 = mesh.indices[i * 3 + 2];
        
        vertexUsed[idx0] = true;
        vertexUsed[idx1] = true;
        vertexUsed[idx2] = true;
        
        indices.append(idx0);
        indices.append(idx1);
        indices.append(idx2);
    }
    
    // Create vertex remapping
    QVector<int> vertexRemap(mesh.vertices.size(), -1);
    QVector<QVector3D> newVertices;
    newVertices.reserve(mesh.vertices.size());
    
    for (int i = 0; i < mesh.vertices.size(); ++i) {
        if (vertexUsed[i]) {
            vertexRemap[i] = newVertices.size();
            newVertices.append(mesh.vertices[i]);
        }
    }
    
    // Remap indices
    QVector<unsigned int> remappedIndices;
    remappedIndices.reserve(indices.size());
    for (unsigned int idx : indices) {
        remappedIndices.append(vertexRemap[idx]);
    }
    
    result.vertices = newVertices;
    result.indices = remappedIndices;
    
    if (removedCount > 0) {
        const int originalTriangles = triangleCount;
        const int remainingTriangles = result.indices.size() / 3;
        qDebug() << "[Simplifier] Removed" << removedCount << "fully internal triangles:" 
                 << originalTriangles << "->" << remainingTriangles;
    }
    
    return result;
}

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

    CleanupStats preCleanupStats;
    MarchingCubes::Mesh workingMesh = cleanupMeshTopology(inputMesh, &preCleanupStats);
    const int workingFaceCount = workingMesh.indices.size() / 3;

    qDebug().noquote() << QString("[Simplifier] request inputVertices=%1 inputFaces=%2 targetFaces=%3 aggressiveness=%4 backend=%5")
                              .arg(inputMesh.vertices.size())
                              .arg(report.inputFaceCount)
                              .arg(targetFaceCount)
                              .arg(aggressiveness, 0, 'f', 2)
                              .arg(backendName());

    qDebug() << "[Simplifier] cleanup pre-pass: vertices" << inputMesh.vertices.size() << "->" << workingMesh.vertices.size()
             << ", faces" << report.inputFaceCount << "->" << workingFaceCount
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

    /*
    Remove useless hidden triangles first,
    then simplify only the meaningful anatomy surface.
    */
    // **NEW: Smart triangle classification to preserve medically important cavity boundaries**
    // Classify triangles into External, CavityBoundary, and FullyInternal
    const QVector<TriangleType> triangleClassification = classifyTrianglesByWindingNumber(workingMesh);
    
    // Remove fully internal triangles (those completely enclosed within the mesh)
    // These don't define any anatomical surface and can be safely removed
    MarchingCubes::Mesh meshWithoutInternal = removeFullyInternalTriangles(workingMesh, triangleClassification);
    const int facesAfterInternalRemoval = meshWithoutInternal.indices.size() / 3;
    
    qDebug() << "[Simplifier] After removing fully internal triangles:" 
             << workingFaceCount << "->" << facesAfterInternalRemoval;
    
    // Use the mesh without internal triangles for further simplification
    MarchingCubes::Mesh simplificationInput = (!meshWithoutInternal.vertices.isEmpty() && !meshWithoutInternal.indices.isEmpty())
        ? meshWithoutInternal
        : workingMesh;

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
    OpenMeshType mesh = toOpenMesh(simplificationInput);
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

    const bool decimationOk = decimater.decimate_to_faces(static_cast<unsigned int>(effectiveTarget));
    qDebug() << "[Simplifier] decimation finished ok=" << decimationOk;
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

    qDebug() << "[Simplifier] input faces:" << report.inputFaceCount
             << "target faces:" << effectiveTarget
             << "output faces:" << report.outputFaceCount
             << "backend:" << backendName();
    return report;
#endif
}

MarchingCubes::Mesh simplifyMesh(const MarchingCubes::Mesh& inputMesh, int targetFaceCount, double aggressiveness)
{
    return simplifyMeshDetailed(inputMesh, targetFaceCount, aggressiveness).mesh;
}

} // namespace MeshSimplifier
#ifndef MARCHINGCUBES_H
#define MARCHINGCUBES_H

#include <QVector>
#include <QVector3D>
#include <QOpenGLFunctions>

class MarchingCubes {
public:
    struct Mesh {
        QVector<QVector3D> vertices;
        QVector<unsigned int> indices;
    };
    static QVector3D interpolateVertex(float isoLevel, int x, int y, int z, float v1, float v2, int edge);
    static Mesh generateMesh(
        const QVector<QVector<QVector<float>>>& volume,
        float isoLevel,
        float spacingX = 1.0f,
        float spacingY = 1.0f,
        float spacingZ = 1.0f
    );
    static Mesh generateMeshStreaming(
        const QVector<QVector<QVector<float>>>& volume,
        float isoLevel,
        int slabDepth = 8,
        float spacingX = 1.0f,
        float spacingY = 1.0f,
        float spacingZ = 1.0f
    );

private:
    static const int edgeTable[256];
    static const int triTable[256][16];

    static float interpolate(float isoLevel, float v1, float v2, float val1, float val2);
};

#endif // MARCHINGCUBES_H

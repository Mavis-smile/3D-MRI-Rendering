#include "GLWidget.h"
#include <QFile>
#include <QDebug>
#include <QTextStream>
#include <QtConcurrent/QtConcurrent>


GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent),
    vbo(QOpenGLBuffer::VertexBuffer),
    ibo(QOpenGLBuffer::IndexBuffer) {
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    zoom = 1.0f;
    cameraPosition = QVector3D(0, 0, 0);

    connect(&meshExportWatcher, &QFutureWatcher<void>::finished,
            this, &GLWidget::startQueuedMeshExport);
}

GLWidget::~GLWidget() {
    meshExportWatcher.waitForFinished();
    makeCurrent();
    glDeleteProgram(shaderProgram); // Proper OpenGL cleanup
    vao.destroy();
    vbo.destroy();
    ibo.destroy();
    doneCurrent();
}

void GLWidget::initializeGL() {
    qDebug() << "Initializing OpenGL context";
    initializeOpenGLFunctions();

    // Check OpenGL limits
    GLint maxVertices, maxIndices;
    glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &maxVertices);
    glGetIntegerv(GL_MAX_ELEMENTS_INDICES, &maxIndices);
    qDebug() << "OpenGL Limits - Max Vertices:" << maxVertices << "Max Indices:" << maxIndices;

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Check OpenGL version
    qDebug() << "OpenGL Version:" << QString((const char*)glGetString(GL_VERSION));
    qDebug() << "GLSL Version:" << QString((const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    createShaders();
    createGeometry();
    // createTestCube();
    // createTestTriangle();

}


void GLWidget::createShaders() {
    // Load vertex shader
    QFile vertFile(":/shaders/shader.vert");
    if (!vertFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCritical() << "Failed to open vertex shader file.";
        return;
    }
    QByteArray vertSource = vertFile.readAll();
    const char* vertShaderSource = vertSource.constData();

    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertShaderSource, nullptr);
    glCompileShader(vertexShader);

    // Check vertex shader compilation
    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        qCritical() << "Vertex shader compilation failed:" << infoLog;
        return;
    }

    // Load fragment shader
    QFile fragFile(":/shaders/shader.frag");
    if (!fragFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qCritical() << "Failed to open fragment shader file.";
        return;
    }
    QByteArray fragSource = fragFile.readAll();
    const char* fragShaderSource = fragSource.constData();

    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Check fragment shader compilation
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        qCritical() << "Fragment shader compilation failed:" << infoLog;
        return;
    }

    // Link shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check linking
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        qCritical() << "Shader program linking failed:" << infoLog;
        return;
    }

    // Cleanup shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Get uniform locations
    modelLoc = glGetUniformLocation(shaderProgram, "model");
    viewLoc = glGetUniformLocation(shaderProgram, "view");
    projLoc = glGetUniformLocation(shaderProgram, "projection");

    qDebug() << "Shader program ID:" << shaderProgram;
    qDebug() << "Uniform locations - model:" << modelLoc << "view:" << viewLoc << "projection:" << projLoc;
}

void GLWidget::createGeometry() {
    vao.create();
    vao.bind();

    vbo.create();
    ibo.create();
}

void GLWidget::updateMesh(const QVector<QVector3D>& vertices, const QVector<unsigned int>& indices) {
    qDebug() << "Updating mesh with" << vertices.size() << "vertices and" << indices.size() << "indices";

    if (!isValid()) {
        qWarning() << "GLWidget context is not valid yet; mesh upload may be deferred until widget is shown.";
    }

    // Debug: Print first 10 vertices
    for (int i = 0; i < qMin(10, vertices.size()); ++i) {
        qDebug() << "Vertex" << i << ":" << vertices[i];
    }

    // Debug: Print first 10 indices
    for (int i = 0; i < qMin(10, indices.size()); ++i) {
        qDebug() << "Index" << i << ":" << indices[i];
    }

    makeCurrent();

    // Delete existing buffers if they exist
    if (vbo.isCreated()) vbo.destroy();
    if (ibo.isCreated()) ibo.destroy();

    vao.bind();

    QVector<QVector3D> normals(vertices.size(), QVector3D(0.0f, 0.0f, 0.0f));
    for (int i = 0; i + 2 < indices.size(); i += 3) {
        const unsigned int ia = indices[i];
        const unsigned int ib = indices[i + 1];
        const unsigned int ic = indices[i + 2];
        if (ia >= static_cast<unsigned int>(vertices.size()) ||
            ib >= static_cast<unsigned int>(vertices.size()) ||
            ic >= static_cast<unsigned int>(vertices.size())) {
            continue;
        }

        const QVector3D& v0 = vertices[ia];
        const QVector3D& v1 = vertices[ib];
        const QVector3D& v2 = vertices[ic];
        QVector3D faceNormal = QVector3D::crossProduct(v1 - v0, v2 - v0);
        if (faceNormal.lengthSquared() <= 1e-12f) {
            continue;
        }

        normals[ia] += faceNormal;
        normals[ib] += faceNormal;
        normals[ic] += faceNormal;
    }

    QVector<float> interleaved;
    interleaved.reserve(vertices.size() * 6);
    for (int i = 0; i < vertices.size(); ++i) {
        QVector3D n = normals[i];
        if (n.lengthSquared() <= 1e-12f) {
            n = QVector3D(0.0f, 0.0f, 1.0f);
        } else {
            n.normalize();
        }

        interleaved.append(vertices[i].x());
        interleaved.append(vertices[i].y());
        interleaved.append(vertices[i].z());
        interleaved.append(n.x());
        interleaved.append(n.y());
        interleaved.append(n.z());
    }

    // Upload vertex data
    vbo.create();
    vbo.bind();
    vbo.allocate(interleaved.constData(), static_cast<int>(interleaved.size() * sizeof(float)));
    qDebug() << "Vertex buffer size:" << vbo.size() << "bytes";

    const GLsizei stride = static_cast<GLsizei>(6 * sizeof(float));
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Upload index data
    ibo.create();
    ibo.bind();
    ibo.allocate(indices.constData(), static_cast<int>(indices.size() * sizeof(unsigned int)));
    qDebug() << "Index buffer size:" << ibo.size() << "bytes";

    indexCount = static_cast<size_t>(indices.size());

    // Calculate mesh bounds
    if (!vertices.isEmpty()) {
        QVector3D min = vertices[0];
        QVector3D max = vertices[0];
        for (const QVector3D& v : vertices) {
            min = QVector3D(qMin(min.x(), v.x()), qMin(min.y(), v.y()), qMin(min.z(), v.z()));
            max = QVector3D(qMax(max.x(), v.x()), qMax(max.y(), v.y()), qMax(max.z(), v.z()));
        }
        meshSize = qMax(qMax(max.x() - min.x(), max.y() - min.y()), max.z() - min.z());
        meshCenter = (min + max) * 0.5f;

        qDebug() << "Mesh size:" << meshSize << "Center:" << meshCenter;
    }

    // Check for OpenGL errors
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        qCritical() << "OpenGL Error during buffer upload:" << err;
    }

    // Always export mesh, but do file I/O in a background thread.
    queueMeshExport(indices, vertices);

    doneCurrent();
    update();
    emit meshUpdateComplete(); // Notify when OpenGL operations are done
}

void GLWidget::paintGL() {
    makeCurrent();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (indexCount == 0) {
        qDebug() << "No indices to draw.";
        return;
    }

    // Model matrix: Center and scale the mesh
    QMatrix4x4 model;
    model.translate(-meshCenter);
    float scaleFactor = 1.0f; // Adjust scaling as needed
    model.scale(scaleFactor);

    // View matrix: Position the camera
    QMatrix4x4 view;
    view.translate(0, 0, -meshSize * zoom); // Apply zoom
    view.rotate(xRot, 1, 0, 0); // Apply rotation
    view.rotate(yRot, 0, 1, 0);
    view.translate(cameraPosition); // Apply panning (after rotation and zoom)

    // Projection matrix: Use a fixed FOV and larger far plane
    QMatrix4x4 projection;
    float nearPlane = 0.1f; // Near clipping plane
    float farPlane = 10000.0f; // Far clipping plane (increased)
    projection.perspective(45.0f, width() / float(height()), nearPlane, farPlane);

    // Use shader program
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model.data());
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view.data());
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection.data());

    // Bind VAO and draw
    vao.bind();
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount), GL_UNSIGNED_INT, 0);

    // Check for OpenGL errors
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        qDebug() << "OpenGL error:" << err;
    }
}

void GLWidget::resizeGL(int w, int h) {
    Q_UNUSED(w);
    Q_UNUSED(h);
    // Handled in paintGL
}

// Mouse handling remains the same as previous implementation
void GLWidget::mousePressEvent(QMouseEvent *event) {
    lastMousePos = event->pos();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event) {
    int dx = event->x() - lastMousePos.x();
    int dy = event->y() - lastMousePos.y();

    if (event->buttons() & Qt::LeftButton) {
        xRot += dy * 0.5f;
        yRot += dx * 0.5f;
        update();
    }

    lastMousePos = event->pos();
}

void GLWidget::wheelEvent(QWheelEvent* event) {
    // Get the scroll amount (positive for scrolling up, negative for scrolling down)
    int delta = event->angleDelta().y();

    // Adjust the zoom factor
    float zoomFactor = 1.1f; // Zoom speed (10% per scroll step)
    if (delta > 0) {
        zoom *= zoomFactor; // Zoom in
    } else if (delta < 0) {
        zoom /= zoomFactor; // Zoom out
    }

    // Clamp the zoom factor to reasonable limits
    zoom = qBound(0.1f, zoom, 10.0f); // Prevent extreme zooming

    // Trigger a repaint
    update();
}

void GLWidget::createTestCube() {
    // Cube vertices (8 corners)
    QVector<QVector3D> vertices = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}
    };

    // Cube indices (12 triangles)
    QVector<unsigned int> indices = {
        0, 1, 2, 2, 3, 0, // Front face
        4, 5, 6, 6, 7, 4, // Back face
        0, 3, 7, 7, 4, 0, // Left face
        1, 2, 6, 6, 5, 1, // Right face
        0, 1, 5, 5, 4, 0, // Bottom face
        2, 3, 7, 7, 6, 2  // Top face
    };

    // Initialize mesh bounds
    meshSize = 2.0f; // Explicitly set for unit cube
    meshCenter = QVector3D(0, 0, 0);

    updateMesh(vertices, indices);
}

void GLWidget::keyPressEvent(QKeyEvent* event) {
    qDebug() << "Key pressed:" << event->key(); // Debug statement
    float panSpeed = 10.0f; // Adjust panning speed as needed

    switch (event->key()) {
    case Qt::Key_Up:
        cameraPosition.setY(cameraPosition.y() - panSpeed); // Move camera up
        break;
    case Qt::Key_Down:
        cameraPosition.setY(cameraPosition.y() + panSpeed); // Move camera down
        break;
    case Qt::Key_Left:
        cameraPosition.setX(cameraPosition.x() + panSpeed); // Move camera left
        break;
    case Qt::Key_Right:
        cameraPosition.setX(cameraPosition.x() - panSpeed); // Move camera right
        break;
    default:
        QOpenGLWidget::keyPressEvent(event); // Pass unhandled keys to the base class
        return;
    }

    qDebug() << "Camera position:" << cameraPosition; // Debug statement
    update(); // Trigger a repaint
}

void GLWidget::createTestTriangle() {
    QVector<QVector3D> vertices = {
        {-1, -1, 0}, {1, -1, 0}, {0, 1, 0}
    };

    QVector<unsigned int> indices = {0, 1, 2};

    updateMesh(vertices, indices);
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent*) {
    resetView();
}

void GLWidget::queueMeshExport(const QVector<unsigned int>& indices, const QVector<QVector3D>& vertices) {
    queuedExportIndices = indices;
    queuedExportVertices = vertices;
    exportQueued = true;
    startQueuedMeshExport();
}

void GLWidget::startQueuedMeshExport() {
    if (meshExportWatcher.isRunning() || !exportQueued) {
        return;
    }

    QVector<unsigned int> indicesToWrite = std::move(queuedExportIndices);
    QVector<QVector3D> verticesToWrite = std::move(queuedExportVertices);
    exportQueued = false;

    meshExportWatcher.setFuture(QtConcurrent::run(
        [indices = std::move(indicesToWrite), vertices = std::move(verticesToWrite)]() mutable {
            GLWidget::exportMeshToFiles(indices, vertices);
        }
    ));
}


void GLWidget::exportMeshToFiles(const QVector<unsigned int>& indices, const QVector<QVector3D>& vertices) {
    // Export vertices (unchanged)
    QFile verticesFile("vertices.txt");
    if (verticesFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&verticesFile);
        for (const QVector3D& v : vertices) {
            out << v.x() << " " << v.y() << " " << v.z() << "\n";
        }
        verticesFile.close();
    }

    // Export faces with validation
    QFile facesFile("faces.txt");
    if (facesFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&facesFile);
        for (int i = 0; i + 2 < indices.size(); i += 3) {
            // Skip invalid faces
            if (indices[i] >= vertices.size() ||
                indices[i+1] >= vertices.size() ||
                indices[i+2] >= vertices.size()) {
                qWarning() << "Skipping invalid face at index" << i;
                continue;
            }
            out << indices[i] << " " << indices[i+1] << " " << indices[i+2] << "\n";
        }
        facesFile.close();
    }
}

void GLWidget::loadMeshFromFiles(const QString& verticesPath, const QString& facesPath) {
    QVector<QVector3D> vertices;
    QVector<unsigned int> indices;

    // Load vertices (unchanged)
    QFile verticesFile(verticesPath);
    if (verticesFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&verticesFile);
        while (!in.atEnd()) {
            QString line = in.readLine().trimmed();
            QStringList parts = line.split(" ", Qt::SkipEmptyParts);
            if (parts.size() == 3) {
                float x = parts[0].toFloat();
                float y = parts[1].toFloat();
                float z = parts[2].toFloat();
                vertices.append(QVector3D(x, y, z));
            }
        }
        verticesFile.close();
    }

    // Load faces with auto-detection
    QFile facesFile(facesPath);
    if (facesFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&facesFile);
        bool isMatlabFormat = false;
        bool formatDetermined = false;

        while (!in.atEnd()) {
            QString line = in.readLine().trimmed();
            QStringList parts = line.split(" ", Qt::SkipEmptyParts);
            if (parts.size() == 3) {
                bool ok1, ok2, ok3;
                unsigned int a = parts[0].toUInt(&ok1);
                unsigned int b = parts[1].toUInt(&ok2);
                unsigned int c = parts[2].toUInt(&ok3);

                if (!ok1 || !ok2 || !ok3) continue;

                // Auto-detect format on first valid line
                if (!formatDetermined) {
                    isMatlabFormat = (a > 0 && b > 0 && c > 0); // MATLAB uses 1-based
                    formatDetermined = true;
                    qDebug() << "Detected file format:" << (isMatlabFormat ? "MATLAB (1-based)" : "Qt (0-based)");
                }

                // Convert indices if MATLAB format
                if (isMatlabFormat) {
                    a--; b--; c--;
                }

                // Validate indices
                if (a < vertices.size() && b < vertices.size() && c < vertices.size()) {
                    indices.append(a);
                    indices.append(b);
                    indices.append(c);
                } else {
                    qWarning() << "Skipping invalid face:" << line
                               << "(vertex count:" << vertices.size() << ")";
                }
            }
        }
        facesFile.close();
    }

    updateMesh(vertices, indices);
}

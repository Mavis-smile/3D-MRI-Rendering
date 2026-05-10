#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aColor;  // Per-vertex color for material visualization

out vec3 vViewPos;
out vec3 vViewNormal;
out vec4 vColor;  // Pass color to fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);
    vec4 viewPos = view * worldPos;
    mat3 normalMat = mat3(transpose(inverse(view * model)));

    vViewPos = viewPos.xyz;
    vViewNormal = normalize(normalMat * aNormal);
    vColor = aColor;  // Pass color to fragment shader
    gl_Position = projection * viewPos;
}

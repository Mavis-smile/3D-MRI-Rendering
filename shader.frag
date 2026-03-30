#version 330 core

in vec3 vWorldPos;
in vec3 vNormal;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(0.45, 0.8, 0.25));
    vec3 V = normalize(-vWorldPos);
    vec3 H = normalize(L + V);

    float diffuse = max(dot(N, L), 0.0);
    float ambient = 0.22;
    float specular = 0.18 * pow(max(dot(N, H), 0.0), 24.0);

    vec3 baseColor = vec3(0.92, 0.90, 0.86);
    vec3 color = baseColor * (ambient + diffuse) + vec3(specular);
    FragColor = vec4(color, 1.0);
}

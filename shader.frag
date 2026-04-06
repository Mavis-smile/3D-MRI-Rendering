#version 330 core

in vec3 vViewPos;
in vec3 vViewNormal;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vViewNormal);
    vec3 V = normalize(-vViewPos);

    vec3 keyLight = normalize(vec3(0.45, 0.78, 0.32));
    vec3 fillLight = normalize(vec3(-0.40, 0.22, -0.62));

    float diffuseKey = max(dot(N, keyLight), 0.0);
    float diffuseFill = max(dot(N, fillLight), 0.0);
    float backFill = max(dot(-N, keyLight), 0.0);

    vec3 H = normalize(keyLight + V);
    float specular = 0.11 * pow(max(dot(N, H), 0.0), 20.0);

    float ambient = 0.22;
    float lightMix = ambient + 0.82 * diffuseKey + 0.24 * diffuseFill + 0.16 * backFill;

    vec3 baseColor = vec3(0.92, 0.91, 0.88);
    vec3 color = baseColor * lightMix + vec3(specular);
    FragColor = vec4(color, 1.0);
}

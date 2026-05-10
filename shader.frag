#version 330 core

in vec3 vViewPos;
in vec3 vViewNormal;
in vec4 vColor;  // Per-vertex material color

out vec4 FragColor;

uniform bool materialColorsEnabled;  // Toggle for material visualization

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
    vec3 shadedBase = baseColor * lightMix + vec3(specular);

    if (materialColorsEnabled && vColor.a > 0.01) {
        float overlayStrength = clamp(vColor.a, 0.0, 1.0);
        vec3 ceramicTint = vColor.rgb;
        vec3 overlayColor = mix(shadedBase, ceramicTint, overlayStrength);
        FragColor = vec4(overlayColor, 1.0);
    } else {
        FragColor = vec4(shadedBase, 1.0);
    }
}

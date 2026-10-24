#version 450
#extension GL_GOOGLE_include_directive : require

layout(binding = 0) uniform sampler2D source;
layout(push_constant) uniform Settings { uvec2 inputSize; uvec2 outputSize; float sharpness; } settings;
layout(location = 0) out vec4 color;

#define A_GPU 1
#define A_GLSL 1
#include "ffx_a.h"
#ifdef FSR_RCAS
#define FSR_RCAS_F 1
AF4 FsrRcasLoadF(ASU2 p) {
    return texelFetch(source, clamp(ivec2(p), ivec2(0), textureSize(source, 0) - 1), 0);
}
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}
#else
#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 p) { return textureGather(source, p, 0); }
AF4 FsrEasuGF(AF2 p) { return textureGather(source, p, 1); }
AF4 FsrEasuBF(AF2 p) { return textureGather(source, p, 2); }
#endif
#include "ffx_fsr1.h"

void main() {
    vec3 rgb;
#ifdef FSR_RCAS
    uvec4 con;
    // UI range 0..1 maps to AMD's two stops..zero stops.
    FsrRcasCon(con, 2.0 * (1.0 - settings.sharpness));
    if (settings.sharpness <= 0.0)
        rgb = FsrRcasLoadF(ivec2(gl_FragCoord.xy)).rgb;
    else
        FsrRcasF(rgb.r, rgb.g, rgb.b, uvec2(gl_FragCoord.xy), con);
#else
    uvec4 c0, c1, c2, c3;
    FsrEasuCon(c0, c1, c2, c3, float(settings.inputSize.x), float(settings.inputSize.y),
               float(settings.inputSize.x), float(settings.inputSize.y),
               float(settings.outputSize.x), float(settings.outputSize.y));
    FsrEasuF(rgb, uvec2(gl_FragCoord.xy), c0, c1, c2, c3);
#endif
    color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}

#version 460
#extension GL_GOOGLE_include_directive : require

#include "../../../Includes/Core.glsl"
#include "../../../Includes/Common/Visibility.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inTangent;
layout(location = 2) in vec2 inUV;
layout(location = 3) in flat uvec3 inId;

layout(location = 0) out vec4 outColor;

void main() {
    // Prompt #8 minimum fragment output. The shader consumes the
    // visibility-format packed id produced by Traditional.vert and
    // writes a cheap diagnostic colour so the swapchain reflects the
    // result of GPU culling + indirect draw. A full deferred
    // material/visibility pass lands in the renderer-validation prompt.
    uint entity = UNPACK_VISIBILITY_ENTITY(inId.x);
    uint payload = inId.z;

    // Cheap hash from the packed entity so every visible instance
    // gets a stable colour per frame. Avoids depending on any
    // bindless texture or material pool read for the baseline pass.
    float r = float((entity * 1664525u + 1013904223u) & 0xFFu) / 255.0;
    float g = float((entity * 22695477u + 1u) & 0xFFu) / 255.0;
    float b = float((payload * 134775813u + 1u) & 0xFFu) / 255.0;

    float ndl = clamp(dot(normalize(inNormal), normalize(vec3(0.3, 1.0, 0.2))), 0.0, 1.0);
    outColor = vec4(r * ndl, g * ndl, b * ndl, 1.0);
}
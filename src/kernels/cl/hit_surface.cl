/*****************************************************************************
 MIT License

 Copyright(c) 2023 Alexander Veselov

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this softwareand associated documentation files(the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions :

 The above copyright noticeand this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *****************************************************************************/

#include "src/kernels/common/shared_structures.h"
#include "src/kernels/common/material.h"
#include "src/kernels/common/sampling.h"
#include "src/kernels/common/light.h"

float getU(HitRecord* record, const __global Triangle* triangles, uint index) {
    Hit hit = record->hits[index];
    Triangle triangle = triangles[hit.primitive_id];

    float u = 0, v = 0;
    float3 v0i = triangle.src.src;
    float3 v0j = triangle.src.dest;
    float3 v1i = triangle.dst.src;
    float3 v1j = triangle.dst.dest;

    float3 x = InterpolateAttributes(triangle.v1.position,
        triangle.v2.position, triangle.v3.position, hit.bc);
    float tan1 = length(cross(v0i - x, v0j - x)) / dot(v0i - x, v0j - x);
    float tan2 = length(cross(v0j - x, v1j - x)) / dot(v0j - x, v1j - x);
    float tan3 = length(cross(v1j - x, v1i - x)) / dot(v1j - x, v1i - x);
    float tan4 = length(cross(v1i - x, v0j - x)) / dot(v1i - x, v0j - x);
    float w0i = (tan1 + tan2) / length(v0i - x);
    float w0j = (tan2 + tan3) / length(v0j - x);
    float w1j = (tan3 + tan4) / length(v1j - x);
    float w1i = (tan4 + tan1) / length(v1i - x);

    u = (w1i + w1j) / (w0i + w0j + w1i + w1j);
    return u;
}

float calcRadiance(HitRecord* record, const __global Triangle* triangles, uint index) {
    float res = getU(record, triangles, index + 1) - getU(record, triangles, index);
    return res;
}

__kernel void HitSurface
(
    // Input
    __global Ray* incoming_rays,
    __global uint* incoming_ray_counter,
    __global uint* incoming_pixel_indices,
    __global Hit* hits,
    __global Triangle* triangles,
    __global Light* analytic_lights,
    __global uint* emissive_indices,
    __global PackedMaterial* materials,
    __global Texture* textures,
    __global uint* texture_data,
    uint bounce,
    uint width,
    uint height,
    __global uint* sample_counter,
    SceneInfo scene_info,
    // Blue noise sampler
    __global int* sobol_256spp_256d,
    __global int* scramblingTile,
    __global int* rankingTile,
    // Output
    __global float3* throughputs,
    __global Ray* outgoing_rays,
    __global uint* outgoing_ray_counter,
    __global uint* outgoing_pixel_indices,
    __global Ray* shadow_rays,
    __global uint* shadow_ray_counter,
    __global uint* shadow_pixel_indices,
    __global float3* direct_light_samples,
    __global float4* result_radiance,

    __global HitRecord* records_buffer
) {
    uint incoming_ray_idx = get_global_id(0);
    uint num_incoming_rays = incoming_ray_counter[0];

    if (incoming_ray_idx >= num_incoming_rays) {
        return;
    }

    Hit hit = hits[incoming_ray_idx];
    HitRecord record = records_buffer[incoming_ray_idx];

    if (record.num <= 0) {
        return;
    }

    //record.num = max(record.num, 0u);

    int shadow_ray_idx = -1;
    int outgoing_ray_idx = -1;
    Ray incoming_ray = incoming_rays[incoming_ray_idx];
    float3 incoming = -incoming_ray.direction.xyz;
    uint pixel_idx = incoming_pixel_indices[incoming_ray_idx];
    uint sample_idx = sample_counter[0];
    int x = pixel_idx % width;
    int y = pixel_idx / width;

    if (hit.primitive_id != INVALID_ID) {

        Triangle triangle = triangles[hit.primitive_id];

        float3 position = InterpolateAttributes(triangle.v1.position,
            triangle.v2.position, triangle.v3.position, hit.bc);

        float3 geometry_normal = normalize(cross(triangle.v2.position - triangle.v1.position, triangle.v3.position - triangle.v1.position));

        float2 texcoord = InterpolateAttributes2(triangle.v1.texcoord.xy,
            triangle.v2.texcoord.xy, triangle.v3.texcoord.xy, hit.bc);

        float3 normal = normalize(InterpolateAttributes(triangle.v1.normal,
            triangle.v2.normal, triangle.v3.normal, hit.bc));

        PackedMaterial packed_material = materials[triangle.mtlIndex];
        Material material;
        ApplyTextures(packed_material, &material, texcoord, textures, texture_data);

        float3 hit_throughput = throughputs[pixel_idx];

        // Direct lighting
        {
            float s_light = SampleRandom(x, y, sample_idx, bounce, SAMPLE_TYPE_LIGHT, BLUE_NOISE_BUFFERS);
            float3 outgoing;
            float pdf;
            float3 light_radiance = Light_Sample(analytic_lights, scene_info, position, normal, s_light, &outgoing, &pdf);

            float distance_to_light = length(outgoing);
            outgoing = normalize(outgoing);

            float3 brdf = EvaluateMaterial(material, normal, incoming, outgoing);
            float3 light_sample = light_radiance * hit_throughput * brdf / pdf * max(dot(outgoing, normal), 0.0f);

            bool spawn_shadow_ray = (pdf > 0.0f) && (dot(light_sample, light_sample) > 0.0f);

            if (spawn_shadow_ray) {
                Ray shadow_ray;
                shadow_ray.origin.xyz = position + normal * EPS;
                shadow_ray.origin.w = 0.0f;
                shadow_ray.direction.xyz = outgoing;
                shadow_ray.direction.w = distance_to_light;

                ///@TODO: use LDS
                shadow_ray_idx = atomic_add(shadow_ray_counter, 1);

                // Store to the memory
                shadow_rays[shadow_ray_idx] = shadow_ray;
                shadow_pixel_indices[shadow_ray_idx] = pixel_idx;
                direct_light_samples[shadow_ray_idx] = light_sample;
            }
        }

        // Indirect lighting
        {
            // Sample bxdf
            float2 s;
            s.x = SampleRandom(x, y, sample_idx, bounce, SAMPLE_TYPE_BXDF_U, BLUE_NOISE_BUFFERS);
            s.y = SampleRandom(x, y, sample_idx, bounce, SAMPLE_TYPE_BXDF_V, BLUE_NOISE_BUFFERS);
            float s1 = SampleRandom(x, y, sample_idx, bounce, SAMPLE_TYPE_BXDF_LAYER, BLUE_NOISE_BUFFERS);

            float pdf = 0.0f;
            float3 throughput = 0.0f;
            float3 outgoing;
            float offset;
            float3 bxdf = SampleBxdf(s1, s, material, normal, incoming, &outgoing, &pdf, &offset);

            if (pdf > 0.0) {
                throughput = bxdf / pdf;
            }

            throughputs[pixel_idx] *= throughput;

            bool spawn_outgoing_ray = (pdf > 0.0);

            if (spawn_outgoing_ray) {
                ///@TODO: use LDS
                outgoing_ray_idx = atomic_add(outgoing_ray_counter, 1);

                Ray outgoing_ray;
                outgoing_ray.origin.xyz = position + geometry_normal * EPS * offset;
                outgoing_ray.origin.w = 0.0f;
                outgoing_ray.direction.xyz = outgoing;
                outgoing_ray.direction.w = MAX_RENDER_DIST;

                outgoing_rays[outgoing_ray_idx] = outgoing_ray;
                outgoing_pixel_indices[outgoing_ray_idx] = pixel_idx;
            }
        }
    }  
    
    bool flag = 0;

    if (shadow_ray_idx == -1) {
        if (record.num != 0) {
            uint pixel_idx = incoming_pixel_indices[incoming_ray_idx];
            shadow_ray_idx = atomic_add(shadow_ray_counter, 1);
            shadow_pixel_indices[shadow_ray_idx] = pixel_idx;

        } else return;
    }

    if (outgoing_ray_idx == -1) {
        if (record.num != 0) {
            //flag = 1;
            //return;
            uint pixel_idx = incoming_pixel_indices[incoming_ray_idx];
            outgoing_ray_idx = atomic_add(outgoing_ray_counter, 1);
            outgoing_pixel_indices[outgoing_ray_idx] = pixel_idx;
        }
    }

    for (int i = 0; i < record.num; i += 2) {

        Hit hit = record.hits[i];
        Triangle triangle = triangles[hit.primitive_id];
        float radiance_base = (record.hits[i + 1].t - record.hits[i].t) * 0.05;

        float u = 0, v = 0;
        float3 v0i = triangle.src.src;
        float3 v0j = triangle.src.dest;
        float3 v1i = triangle.dst.src;
        float3 v1j = triangle.dst.dest;
    
        float3 x = InterpolateAttributes(triangle.v1.position,
            triangle.v2.position, triangle.v3.position, hit.bc);
        float tan1 = length(cross(v0i - x, v0j - x)) / dot(v0i - x, v0j - x);
        float tan2 = length(cross(v0j - x, v1j - x)) / dot(v0j - x, v1j - x);
        float tan3 = length(cross(v1j - x, v1i - x)) / dot(v1j - x, v1i - x);
        float tan4 = length(cross(v1i - x, v0j - x)) / dot(v1i - x, v0j - x);
        float w0i = (tan1 + tan2) / length(v0i - x);
        float w0j = (tan2 + tan3) / length(v0j - x);
        float w1j = (tan3 + tan4) / length(v1j - x);
        float w1i = (tan4 + tan1) / length(v1i - x);

        u = (w1i + w1j) / (w0i + w0j + w1i + w1j);
        v = (w0i + w0j) / (w0i + w0j + w1i + w1j);

        float3 p = (1.0f - u) * ((1.0f - v) * v0i + v * v0j) + u * ((1.0f - v) * v1i + v * v1j);

        triangle = triangles[triangle.src.tri1];
        
        float radiance = calcRadiance(&record, triangles, i);

        //// Rendering
        {
            Ray incoming_ray = incoming_rays[incoming_ray_idx];
            float3 incoming = -incoming_ray.direction.xyz;

            uint pixel_idx = incoming_pixel_indices[incoming_ray_idx];
            uint sample_idx = sample_counter[0];

            int x = pixel_idx % width;
            int y = pixel_idx / width;

            /*float3 position = InterpolateAttributes(triangle.v1.position,
                triangle.v2.position, triangle.v3.position, (0.5, 0.5));*/

            float3 position = p;

            float3 geometry_normal = normalize(cross(triangle.v2.position - triangle.v1.position, triangle.v3.position - triangle.v1.position));

            float2 texcoord = InterpolateAttributes2(triangle.v1.texcoord.xy,
                triangle.v2.texcoord.xy, triangle.v3.texcoord.xy, (0.5, 0.5));

            float3 normal = normalize(InterpolateAttributes(triangle.v1.normal,
                triangle.v2.normal, triangle.v3.normal, (0.5, 0.5)));

            PackedMaterial packed_material = materials[triangle.mtlIndex];
            Material material;
            ApplyTextures(packed_material, &material, texcoord, textures, texture_data);

            float3 hit_throughput = throughputs[pixel_idx];

           

            // Direct lighting
            {
                float s_light = SampleRandom(x, y, sample_idx, bounce, SAMPLE_TYPE_LIGHT, BLUE_NOISE_BUFFERS);
                float3 outgoing;
                float pdf;
                float3 light_radiance = Light_Sample(analytic_lights, scene_info, position, normal, s_light, &outgoing, &pdf);

                float distance_to_light = length(outgoing);
                outgoing = normalize(outgoing);

                float3 brdf = EvaluateMaterial(material, normal, incoming, outgoing);
                float3 light_sample = light_radiance * hit_throughput * brdf / pdf * max(dot(outgoing, normal), 0.0f);
                light_sample.x = max(light_sample.x, 0.0f);
                light_sample.y = max(light_sample.y, 0.0f);
                light_sample.z = max(light_sample.z, 0.0f);


                bool spawn_shadow_ray = (pdf > 0.0f) && (dot(light_sample, light_sample) > 0.0f);

                if (spawn_shadow_ray) {
                    Ray shadow_ray;
                    shadow_ray.origin.xyz = position + normal * EPS;
                    shadow_ray.origin.w = 0.0f;
                    shadow_ray.direction.xyz = outgoing;
                    shadow_ray.direction.w = distance_to_light;

                    ///@TODO: use LDS

                    // Store to the memory
                    //shadow_rays[shadow_ray_idx] = shadow_ray;
                    // shadow_pixel_indices[shadow_ray_idx] = pixel_idx;
                    direct_light_samples[shadow_ray_idx] += light_sample * radiance;
                    //float3 var = light_sample;
                    //direct_light_samples[shadow_ray_idx] = direct_light_samples[shadow_ray_idx] + (1.0f, 1.0f, 1.0f);
                    //direct_light_samples[shadow_ray_idx] = min(direct_light_samples[shadow_ray_idx], (0.0f, 0.0f, 0.0f));
                    //direct_light_samples[shadow_ray_idx] -= var;
                  
                }
            }

            

            // Indirect lighting
            {
                // Sample bxdf
                float2 s;
                s.x = SampleRandom(x, y, sample_idx, bounce, SAMPLE_TYPE_BXDF_U, BLUE_NOISE_BUFFERS);
                s.y = SampleRandom(x, y, sample_idx, bounce, SAMPLE_TYPE_BXDF_V, BLUE_NOISE_BUFFERS);
                float s1 = SampleRandom(x, y, sample_idx, bounce, SAMPLE_TYPE_BXDF_LAYER, BLUE_NOISE_BUFFERS);

                float pdf = 0.0f;
                float3 throughput = 0.0f;
                float3 outgoing;
                float offset;
                float3 bxdf = SampleBxdf(s1, s, material, normal, incoming, &outgoing, &pdf, &offset);

                if (pdf > 0.0) {
                    throughput = bxdf / pdf;
                }

                throughputs[pixel_idx] *= throughput;

                bool spawn_outgoing_ray = (pdf > 0.0);

                if (spawn_outgoing_ray) {
                    ///@TODO: use LDS
                    

                    Ray outgoing_ray;
                    outgoing_ray.origin.xyz = position + geometry_normal * EPS * offset;
                    outgoing_ray.origin.w = 0.0f;
                    outgoing_ray.direction.xyz = outgoing;
                    outgoing_ray.direction.w = MAX_RENDER_DIST;

                    // outgoing_rays[outgoing_ray_idx] = outgoing_ray;
                    outgoing_pixel_indices[outgoing_ray_idx] = pixel_idx;
                }
            }
        }

    }

    

}

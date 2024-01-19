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

float getU(Hit hit, const __global Triangle* triangles, uint pixel_idx) {
    Triangle triangle = triangles[hit.primitive_id];

    float u = 0, v = 0;
    float3 v0i = triangle.src.src;
    float3 v0j = triangle.src.dest;
    float3 v1i = triangle.dst.src;
    float3 v1j = triangle.dst.dest;

    float3 x = InterpolateAttributes(triangle.v1.position,
        triangle.v2.position, triangle.v3.position, hit.bc);
    float tan1 = ((length(v0i - x) * length(v0j - x)) - dot(v0i - x, v0j - x)) / length(cross(v0i - x, v0j - x));
    float tan2 = ((length(v0j - x) * length(v1j - x)) - dot(v0j - x, v1j - x)) / length(cross(v0j - x, v1j - x));
    float tan3 = ((length(v1j - x) * length(v1i - x)) - dot(v1j - x, v1i - x)) / length(cross(v1j - x, v1i - x));
    float tan4 = ((length(v1i - x) * length(v0i - x)) - dot(v1i - x, v0i - x)) / length(cross(v1i - x, v0i - x));
    float w0i = (tan1 + tan4) / length(v0i - x);
    float w0j = (tan1 + tan2) / length(v0j - x);
    float w1j = (tan2 + tan3) / length(v1j - x);
    float w1i = (tan4 + tan3) / length(v1i - x);
    float tot = w0i + w0j + w1j + w1i;
    w0i /= tot, w0j /= tot, w1j /= tot, w1i /= tot;
    u = (w1i + w1j) / (w0i + w0j + w1i + w1j);
    //if (pixel_idx == 0) {
    //    printf("Bilinear Coordiates: %.2lf %.2lf %.2lf %.2lf\n", w0i, w0j, w1i, w1j);
    //    printf("XYZU: %.2lf %.2lf\n", x.z, u);
    //}
    //return x.z;
    return u;
}

bool compare(Hit a, Hit b) {
    if (a.exact_id != b.exact_id) {
        return a.exact_id < b.exact_id;
    }

    // TODO: Fix occlusion culling problem

    return a.time < b.time;
}

struct Interval {
    float l, r;
    float padding[2];
};

bool compare2(struct Interval a, struct Interval b) {
    if (a.l != b.l) {
        return a.l < b.l;
    }
    return a.r < b.r;
}

void quickSort(Hit* a, int l, int r) {
    if (l < r) {
        int i = l, j = r;
        Hit x = a[l];
        while (i < j) {
            while (i < j && !compare(a[j], x)) j--;
            if (i < j) a[i++] = a[j];
            while (i < j && compare(a[i], x)) i++;
            if (i < j) a[j--] = a[i];
        }
        a[i] = x;
        quickSort(a, l, i - 1);
        quickSort(a, i + 1, r);
    }
}

void sort(__global HitRecord* record) {
    int n = min(30u, record->num);
    Hit* a = record->hits;
    quickSort(a, 0, n - 1);
}

void quickSort2(struct Interval* a, int l, int r) {
    if (l < r) {
        int i = l, j = r;
        struct Interval x = a[l];
        while (i < j) {
            while (i < j && !compare2(a[j], x)) j--;
            if (i < j) a[i++] = a[j];
            while (i < j && compare2(a[i], x)) i++;
            if (i < j) a[j--] = a[i];
        }
        a[i] = x;
        quickSort2(a, l, i - 1);
        quickSort2(a, i + 1, r);
    }
}

float minfloat2(float2 x) {
    return min(x.x, x.y);
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

    if (records_buffer[incoming_ray_idx].num <= 1) {
        return;
    }
    uint pixel_idx = incoming_pixel_indices[incoming_ray_idx];
    
    uint sample_idx = sample_counter[0];

    /*if (pixel_idx == 0) {
        if (records_buffer[incoming_ray_idx].num & 1) {
            printf("Error: %d\n", records_buffer[incoming_ray_idx].num);
        }
    }*/

    {
        
        for (int i = 0, limit = min(30u, records_buffer[incoming_ray_idx].num); i < limit; ++i) {
            Triangle triangle = triangles[records_buffer[incoming_ray_idx].hits[i].primitive_id];
            if ((triangle.prismTri & 1) == 0) {
                if (triangle.prismTri == 0) records_buffer[incoming_ray_idx].hits[i].time = 0;
                else if (triangle.prismTri == 2) records_buffer[incoming_ray_idx].hits[i].time = 1;
            } else {
                float u = getU(records_buffer[incoming_ray_idx].hits[i], triangles, pixel_idx);
                records_buffer[incoming_ray_idx].hits[i].time = u;
            }
        }   

    }

    sort(&records_buffer[incoming_ray_idx]);


    uint shadow_ray_idx = atomic_add(shadow_ray_counter, 1);
    shadow_pixel_indices[shadow_ray_idx] = pixel_idx;
    Hit hit = hits[incoming_ray_idx];
    HitRecord record = records_buffer[incoming_ray_idx];
    direct_light_samples[shadow_ray_idx] = (float3) (0.0f, 0.0f, 0.0f);
    

    {
        int flag = 0;
        //int cnt = 0;
        const float eps = 1e-3;
        for (int i = 0, limit = min(30u, record.num); i < limit; ++i) {
            if (record.hits[i].time < eps || record.hits[i].time + eps >= 1) {
                if (fabs(minfloat2(record.hits[i].bc)) < eps || record.hits[i].bc.x + record.hits[i].bc.y + eps >= 1) {
                    direct_light_samples[shadow_ray_idx] += (float3) (0.1f, 2.1f, 0.1f);
                }
            }
        }


        struct Interval interval[30];
        int cnt = 0;

        for (int i = 0, limit = min(30u, record.num); i + 1 < limit; i += 2) {
            interval[cnt].l = record.hits[i].time;
            interval[cnt].r = record.hits[i + 1].time;
            ++cnt;
            //if (pixel_idx == 0) printf("Origin %d: %f %f\n", cnt, record.hits[i].time, record.hits[i + 1].time);
            //if (pixel_idx == 0) printf("Origin int %d: %f %f\n", cnt, interval[cnt - 1].l, interval[cnt - 1].r);
        }

        if (cnt == 0) return;

        float cur = 0;
        quickSort2(interval, 0, cnt - 1);

        for (int i = 0; i < cnt; ++i) {
            //if (pixel_idx == 0) printf("Dist %d: %f %f\n", i, interval[i].l, interval[i].r);
            if (cur > interval[i].r) continue;
            float radiance = interval[i].r - max(interval[i].l, cur);
            direct_light_samples[shadow_ray_idx] += (float3) (1.0f * radiance, 0.1f * radiance, 0.1f * radiance);
            cur = max(cur, interval[i].r);
        }

    }

    //delete[] interval;


    //if (get_global_id(0) == 0) {
    //    printf("finish\n");
    //}

}

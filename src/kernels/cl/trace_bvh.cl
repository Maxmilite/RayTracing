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
#include "src/kernels/common/constants.h"

//float getU(Hit hit, const __global Triangle* triangles) {
//    Triangle triangle = triangles[hit.primitive_id];
//
//    float u = 0, v = 0;
//    float3 v0i = triangle.src.src;
//    float3 v0j = triangle.src.dest;
//    float3 v1i = triangle.dst.src;
//    float3 v1j = triangle.dst.dest;
//
//    float3 x = InterpolateAttributes(triangle.v1.position,
//        triangle.v2.position, triangle.v3.position, hit.bc);
//    float tan1 = length(cross(v0i - x, v0j - x)) / dot(v0i - x, v0j - x);
//    float tan2 = length(cross(v0j - x, v1j - x)) / dot(v0j - x, v1j - x);
//    float tan3 = length(cross(v1j - x, v1i - x)) / dot(v1j - x, v1i - x);
//    float tan4 = length(cross(v1i - x, v0j - x)) / dot(v1i - x, v0j - x);
//    float w0i = (tan1 + tan2) / length(v0i - x);
//    float w0j = (tan2 + tan3) / length(v0j - x);
//    float w1j = (tan3 + tan4) / length(v1j - x);
//    float w1i = (tan4 + tan1) / length(v1i - x);
//
//    u = (w1i + w1j) / (w0i + w0j + w1i + w1j);
//    return u;
//}

bool RayTriangle(Ray ray, const __global RTTriangle* triangle, float2* bc, float* out_t) {

    float3 e1 = triangle->position2 - triangle->position1;
    float3 e2 = triangle->position3 - triangle->position1;
    // Calculate planes normal vector
    float3 pvec = cross(ray.direction.xyz, e2);
    float det = dot(e1, pvec);

    // Ray is parallel to plane
    if (det < 1e-8f && -det < 1e-8f) {
        return false;
    }
    //if (fabs(det) < 1e-8f) return false;

    float inv_det = 1.0f / det;
    float3 tvec = ray.origin.xyz - triangle->position1;
    float u = dot(tvec, pvec) * inv_det;

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 qvec = cross(tvec, e1);
    float v = dot(ray.direction.xyz, qvec) * inv_det;

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = dot(e2, qvec) * inv_det;
    float t_min = ray.origin.w;
    float t_max = ray.direction.w;

    if (t < t_min || t > t_max) {
        return false;
    }

     //if ((triangle->prismTri & 1) == 1) return false;

    // Intersection is found
    *bc = (float2)(u, v);
    *out_t = t;

    return true;
}

float max3(float3 val) {
    return max(max(val.x, val.y), val.z);
}

float min3(float3 val) {
    return min(min(val.x, val.y), val.z);
}

bool RayBounds(Bounds3 bounds, float3 ray_origin, float3 ray_inv_dir, float t_min, float t_max) {
    float3 aabb_min = bounds.pos[0];
    float3 aabb_max = bounds.pos[1];

    float3 t0 = (aabb_min - ray_origin) * ray_inv_dir;
    float3 t1 = (aabb_max - ray_origin) * ray_inv_dir;

    //if (ray_inv_dir.xyz < 0.0f) { float3 var = t0; t0 = t1; t1 = var; }

    float tmin = max(max3(min(t0, t1)), t_min);
    float tmax = min(min3(max(t0, t1)), t_max);

    return (tmax >= tmin);

    //float tmin = t_min, tmax = t_max;

    /* {
        if (ray_inv_dir.x < 0.0f) { float var = t0.x; t0.x = t1.x; t1.x = var; }
        if (ray_inv_dir.y < 0.0f) { float var = t0.y; t0.y = t1.y; t1.y = var; }
        if (ray_inv_dir.z < 0.0f) { float var = t0.z; t0.z = t1.z; t1.z = var; }
        float tmin = max(max3(min(t0, t1)), t_min);
        float tmax = min(min3(max(t0, t1)), t_max);
        return (tmax >= tmin);
    }*/
    
}

__kernel void TraceBvh
(
    // Input
    __global Ray* rays,
    __global uint* ray_counter,
    __global RTTriangle* triangles,
    __global LinearBVHNode* nodes,
    // Output
    __global uint* shadow_hits,
    __global Hit* hits,
    __global HitRecord* records
) {
    uint ray_idx = get_global_id(0);
    ///@TODO: use indirect dispatch
    uint num_rays = ray_counter[0];

    if (ray_idx >= num_rays) {
        return;
    }

    Ray ray = rays[ray_idx];
    // TODO: fix it
    float3 ray_inv_dir = (float3)(1.0f, 1.0f, 1.0f) / ray.direction.xyz;
    int ray_sign[3];
    ray_sign[0] = ray_inv_dir.x < 0;
    ray_sign[1] = ray_inv_dir.y < 0;
    ray_sign[2] = ray_inv_dir.z < 0;

//#ifdef SHADOW_RAYS
    uint shadow_hit = INVALID_ID;
//#endif

    float t;
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0;
    int currentNodeIndex = 0;
    int nodesToVisit[64];


    records[ray_idx].num = 0;
    while (true) {
        LinearBVHNode node = nodes[currentNodeIndex];
        if (RayBounds(node.bounds, ray.origin.xyz, ray_inv_dir, ray.origin.w, ray.direction.w)) {
            int num_primitives = node.num_primitives_axis >> 16;
            // Leaf node
            if (num_primitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                for (int i = 0; i < num_primitives; ++i) {
                    Hit hit;
                    if (RayTriangle(ray, &triangles[node.offset + i], &hit.bc, &hit.t)) {
                        hit.primitive_id = node.offset + i;
                        // Set ray t_max
                        // TODO: remove t from hit structure
                        ray.direction.w = hit.t;
//#ifdef SHADOW_RAYS
                        shadow_hit = INVALID_ID;
                            //goto endtrace;
//#endif
                        RTTriangle triangle = triangles[node.offset + i];
                        if ((triangle.prismTri & 1) == 0) {
                            hit.exact_id = hit.primitive_id;
                            if (triangle.prismTri == 2) hit.exact_id--;
                            records[ray_idx].hits[records[ray_idx].num] = hit;
                            records[ray_idx].num++;
                        }
                        else if (triangle.src.triCount == 2) {
                            hit.exact_id = triangle.src.tri1;
                            records[ray_idx].hits[records[ray_idx].num] = hit;
                            records[ray_idx].num++;
                            hit.exact_id = triangle.src.tri2;
                            records[ray_idx].hits[records[ray_idx].num] = hit;
                            records[ray_idx].num++;
                        }
                        else {
                            hit.exact_id = triangle.src.tri1;
                            records[ray_idx].hits[records[ray_idx].num] = hit;
                            records[ray_idx].num++;
                        }
                    }
                }

                if (toVisitOffset == 0) {
                    break;
                }

                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else {
                // Put far BVH node on _nodesToVisit_ stack, advance to near node
                if (ray_sign[node.num_primitives_axis & 0xFFFF]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node.offset;
                }
                else {
                    nodesToVisit[toVisitOffset++] = node.offset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        }
        else {
            if (toVisitOffset == 0) {
                break;
            }

            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }

endtrace:
    // Write the result to the output buffer
//#ifdef SHADOW_RAYS
    shadow_hits[ray_idx] = shadow_hit;
 //#else
    do {} while (0);

    // TODO: Remove this

    if (records[ray_idx].num) hits[ray_idx] = records[ray_idx].hits[0];
    else {
        Hit hit;
        hit.primitive_id = INVALID_ID;
        hits[ray_idx] = hit;
    }

}

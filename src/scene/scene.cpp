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

#include <GL/glew.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "scene.hpp"
#include "mathlib/mathlib.hpp"
#include "render.hpp"
#include "utils/cl_exception.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cctype>
#include <functional>
#include <unordered_map>
#include <iomanip>

#undef max

#ifdef __cplusplus
struct RTMatrix {
    float data[4][4];

    enum Axis {
        X, Y, Z
    };

    RTMatrix() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                data[i][j] = (i == j);
            }
        }
    }
    RTMatrix(const float t[4][4]) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                data[i][j] = t[i][j];
            }
        }
    }
    RTMatrix operator* (const RTMatrix& x) {
        RTMatrix res;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                res.data[i][j] = 0;
                for (int k = 0; k < 4; ++k) {
                    res.data[i][j] += this->data[i][k] * x.data[k][j];
                }
            }
        }
        return res;
    }
    RTMatrix& operator*= (const RTMatrix& x) {
        RTMatrix res = (*this) * x;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                this->data[i][j] = res.data[i][j];
            }
        }
        return *this;
    }

    float3 transform(const float3& x) {
        float ori[4] = { x.x, x.y, x.z, 1 }, res[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 4; ++k) {
                res[i] += ori[k] * this->data[k][i];
            }
        }
        return float3(res[0], res[1], res[2]);
    }

    void translation(float3 t) {
        float var[4][4] = {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {t.x, t.y, t.z, 1}
        };
        RTMatrix T = var;
        (*this) *= T;
    }

    void rotation(Axis axis, float deg) {
        RTMatrix T;
        if (axis == Axis::Z) {
            float var[4][4] = {
                {cos(deg), sin(deg), 0, 0},
                {-sin(deg), cos(deg), 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
            };
            T = var;
        }
        else if (axis == Axis::X) {
            float var[4][4] = {
                {1, 0, 0, 0},
                {0, cos(deg), sin(deg), 0},
                {0, -sin(deg), cos(deg), 0},
                {0, 0, 0, 1}
            };
            T = var;
        }
        else if (axis == Axis::Y) {
            float var[4][4] = {
                {cos(deg), -sin(deg), 0, 0},
                {0, 1, 0, 0},
                {sin(deg), cos(deg), 0, 0},
                {0, 0, 0, 1}
            };
            T = var;
        }
        (*this) *= T;
    }
};
#endif

Scene::Scene(const char* filename, float scale, bool flip_yz) {
    Load(filename, scale, flip_yz);
}

namespace {
    unsigned int PackAlbedo(float r, float g, float b, std::uint32_t texture_index) {
        assert(texture_index < 256);
        r = clamp(r, 0.0f, 1.0f);
        g = clamp(g, 0.0f, 1.0f);
        b = clamp(b, 0.0f, 1.0f);
        return ((unsigned int) (r * 255.0f)) | ((unsigned int) (g * 255.0f) << 8)
            | ((unsigned int) (b * 255.0f) << 16) | (texture_index << 24);
    }

    unsigned int PackRGBE(float r, float g, float b) {
        // Make sure the values are not negative
        r = std::max(r, 0.0f);
        g = std::max(g, 0.0f);
        b = std::max(b, 0.0f);

        float v = r;
        if (g > v) v = g;
        if (b > v) v = b;

        if (v < 1e-32f) {
            return 0;
        }
        else {
            int e;
            v = frexp(v, &e) * 256.0f / v;
            return ((unsigned int) (r * v)) | ((unsigned int) (g * v) << 8)
                | ((unsigned int) (b * v) << 16) | ((e + 128) << 24);
        }
    }

    float3 UnpackRGBE(unsigned int rgbe) {
        float f;
        int r = (rgbe >> 0) & 0xFF;
        int g = (rgbe >> 8) & 0xFF;
        int b = (rgbe >> 16) & 0xFF;
        int exp = rgbe >> 24;

        if (exp) {   /*nonzero pixel*/
            f = ldexp(1.0f, exp - (int) (128 + 8));
            return float3((float) r, (float) g, (float) b) * f;
        }
        else {
            return 0.0;
        }
    }

    unsigned int PackRoughnessMetalness(float roughness, std::uint32_t roughness_idx,
        float metalness, std::uint32_t metalness_idx) {
        assert(roughness_idx < 256 && metalness_idx < 256);
        roughness = clamp(roughness, 0.0f, 1.0f);
        metalness = clamp(metalness, 0.0f, 1.0f);
        return ((unsigned int) (roughness * 255.0f)) | (roughness_idx << 8)
            | ((unsigned int) (metalness * 255.0f) << 16) | (metalness_idx << 24);
    }

    unsigned int PackIorEmissionIdxTransparency(float ior, std::uint32_t emission_idx,
        float transparency, std::uint32_t transparency_idx) {
        assert(emission_idx < 256 && transparency_idx < 256);
        ior = clamp(ior, 0.0f, 10.0f);
        transparency = clamp(transparency, 0.0f, 1.0f);
        return ((unsigned int) (ior * 25.5f)) | (emission_idx << 8)
            | ((unsigned int) (transparency * 255.0f) << 16) | (transparency_idx << 24);
    }
}

void Scene::Load(const char* filename, float scale, bool flip_yz) {
    std::cout << "Loading object file " << filename << std::endl;

    std::string path_to_folder = std::string(filename, std::string(filename).rfind('/') + 1);

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, path_to_folder.c_str());

    if (!success) {
        throw std::runtime_error("Failed to load the scene!");
    }

    materials_.resize(materials.size());

    const float kGamma = 2.2f;
    const std::uint32_t kInvalidTextureIndex = 0xFF;

    for (std::uint32_t material_idx = 0; material_idx < materials.size(); ++material_idx) {
        auto& out_material = materials_[material_idx];
        auto const& in_material = materials[material_idx];

        // Convert from sRGB to linear
        out_material.diffuse_albedo = PackAlbedo(
            pow(in_material.diffuse[0], kGamma), // R
            pow(in_material.diffuse[1], kGamma), // G
            pow(in_material.diffuse[2], kGamma), // B
            in_material.diffuse_texname.empty() ? kInvalidTextureIndex :
            LoadTexture((path_to_folder + in_material.diffuse_texname).c_str()));

        out_material.specular_albedo = PackAlbedo(
            pow(in_material.specular[0], kGamma), // R
            pow(in_material.specular[1], kGamma), // G
            pow(in_material.specular[2], kGamma), // B
            in_material.specular_texname.empty() ? kInvalidTextureIndex :
            LoadTexture((path_to_folder + in_material.specular_texname).c_str()));

        out_material.emission = PackRGBE(in_material.emission[0], in_material.emission[1], in_material.emission[2]);

        out_material.roughness_metalness = PackRoughnessMetalness(
            in_material.roughness,
            in_material.roughness_texname.empty() ? kInvalidTextureIndex :
            LoadTexture((path_to_folder + in_material.roughness_texname).c_str()),
            in_material.metallic,
            in_material.metallic_texname.empty() ? kInvalidTextureIndex :
            LoadTexture((path_to_folder + in_material.metallic_texname).c_str()));

        out_material.ior_emission_idx_transparency = PackIorEmissionIdxTransparency(
            in_material.ior, in_material.emissive_texname.empty() ? kInvalidTextureIndex :
            LoadTexture((path_to_folder + in_material.emissive_texname).c_str()),
            in_material.transmittance[0], in_material.alpha_texname.empty() ? kInvalidTextureIndex :
            LoadTexture((path_to_folder + in_material.alpha_texname).c_str()));

    }

    auto flip_vector = [](float3& vec, bool do_flip) {
        if (do_flip) {
            std::swap(vec.y, vec.z);
            vec.y = -vec.y;
        }
        };

    std::map<std::pair<float3, float3>, std::pair<int, int>> umap;

    auto handle_edge = [&](Vertex v1, Vertex v2, std::uint32_t face) {
        float3 pos1 = v1.position, pos2 = v2.position;
        if (pos1 > pos2) {
            std::swap(pos1, pos2);
        }
        if (!umap.count({ pos1, pos2 })) {
            umap[{pos1, pos2}] = { face, -1 };
        }
        else if (umap[{pos1, pos2}].second == -1) {
            umap[{pos1, pos2}].second = face;
        }
        else throw std::runtime_error("Error occurred while handling edge.");
    };

    RTMatrix transform;
    //transform.rotation(transform.Axis::Z, acos(-1) / 12 * 2);
    transform.translation(float3(0.2f, 0.2f, 1.0f));

    for (auto const& shape : shapes) {
        auto const& indices = shape.mesh.indices;
        // The mesh is triangular
        assert(indices.size() % 3 == 0);

        for (std::uint32_t face = 0; face < indices.size() / 3; ++face) {

            int pos_idx[] = {
                indices[face * 3 + 0].vertex_index, 
                indices[face * 3 + 1].vertex_index, 
                indices[face * 3 + 2].vertex_index, 
            };

            int normal_idx[] = {
                indices[face * 3 + 0].normal_index,
                indices[face * 3 + 1].normal_index,
                indices[face * 3 + 2].normal_index,
            };

            int texcoord_idx[] = {
                indices[face * 3 + 0].texcoord_index,
                indices[face * 3 + 1].texcoord_index,
                indices[face * 3 + 2].texcoord_index,
            };

            Vertex v[3];
            for (int i = 0; i < 3; ++i) {
                v[i].position.x = attrib.vertices[pos_idx[i] * 3 + 0] * scale;
                v[i].position.y = attrib.vertices[pos_idx[i] * 3 + 1] * scale;
                v[i].position.z = attrib.vertices[pos_idx[i] * 3 + 2] * scale;

                v[i].normal.x = attrib.normals[normal_idx[i] * 3 + 0];
                v[i].normal.y = attrib.normals[normal_idx[i] * 3 + 1];
                v[i].normal.z = attrib.normals[normal_idx[i] * 3 + 2];

                v[i].texcoord.x = texcoord_idx[i] < 0 ? 0.0f : attrib.texcoords[texcoord_idx[i] * 2 + 0];
                v[i].texcoord.y = texcoord_idx[i] < 0 ? 0.0f : attrib.texcoords[texcoord_idx[i] * 2 + 1];

                flip_vector(v[i].position, flip_yz);
                flip_vector(v[i].normal, flip_yz);
            }


            // Prism
            
            Vertex moved_v[3] = { v[0], v[1], v[2] };
            for (int i = 0; i < 3; ++i) {
                moved_v[i].position = transform.transform(moved_v[i].position);
            }
            

            //for (int i = 0; i < 3; ++i) {
            //    std::cerr << v[i].position.x << " "
            //        << v[i].position.y << " "
            //        << v[i].position.z << std::endl;
            //}

            //for (int i = 0; i < 3; ++i) {
            //    std::cerr << moved_v[i].position.x << " " 
            //        << moved_v[i].position.y << " "
            //        << moved_v[i].position.z << std::endl;
            //}

            triangles_.emplace_back(v[0], v[1], v[2], 0, 0);
            triangles_.back().exact_id = triangles_.size() - 1;
            triangles_.back().origin_idx = triangles_.size() - 1;
            triangles_.emplace_back(moved_v[0], moved_v[1], moved_v[2], 0, 2);
            triangles_.back().exact_id = triangles_.size() - 2;
            triangles_.back().origin_idx = triangles_.size() - 1;

            handle_edge(v[0], v[1], triangles_.size() - 2);
            handle_edge(v[1], v[2], triangles_.size() - 2);
            handle_edge(v[2], v[0], triangles_.size() - 2);
        }

    }

    std::cerr << "Edge size: " << umap.size() << std::endl;

    for (auto& [i, j] : umap) {
        auto& [src, dst] = i;
        if (j.second != -1) {
            edges_.emplace_back(src, dst, j.first, j.second);
        }
        else {
            edges_.emplace_back(src, dst, j.first);
        }
    }

    for (const auto& edge : edges_) {

        auto edgeDst = edge;
        edgeDst.src = transform.transform(edgeDst.src);
        edgeDst.dest = transform.transform(edgeDst.dest);

        Vertex v1;
        v1.position.x = edge.src.x;
        v1.position.y = edge.src.y;
        v1.position.z = edge.src.z;

        Vertex v2;
        v2.position.x = edge.dest.x;
        v2.position.y = edge.dest.y;
        v2.position.z = edge.dest.z;

        Vertex v3;
        v3.position.x = edgeDst.src.x;
        v3.position.y = edgeDst.src.y;
        v3.position.z = edgeDst.src.z;

        flip_vector(v1.position, flip_yz);
        flip_vector(v1.normal, flip_yz);
        flip_vector(v2.position, flip_yz);
        flip_vector(v2.normal, flip_yz);
        flip_vector(v3.position, flip_yz);
        flip_vector(v3.normal, flip_yz);

        Triangle triangle(v1, v2, v3, 0, 1);
        triangle.InsertEdge(edge, edgeDst);
        triangles_.emplace_back(triangle);
        triangles_.back().origin_idx = triangles_.size() - 1;

        v1.position.x = edgeDst.src.x;
        v1.position.y = edgeDst.src.y;
        v1.position.z = edgeDst.src.z;
        v2.position.x = edgeDst.dest.x;
        v2.position.y = edgeDst.dest.y;
        v2.position.z = edgeDst.dest.z;
        v3.position.x = edge.dest.x;
        v3.position.y = edge.dest.y;
        v3.position.z = edge.dest.z;

        //cr = cross(v2.position - v1.position, v3.position - v2.position);
        //v1.normal.x = v2.normal.x = v3.normal.x = cr.x;
        //v1.normal.y = v2.normal.y = v3.normal.y = cr.y;
        //v1.normal.z = v2.normal.z = v3.normal.z = cr.z;

        flip_vector(v1.position, flip_yz);
        flip_vector(v1.normal, flip_yz);
        flip_vector(v2.position, flip_yz);
        flip_vector(v2.normal, flip_yz);
        flip_vector(v3.position, flip_yz);
        flip_vector(v3.normal, flip_yz);

        triangle = Triangle(v1, v2, v3, 0, 3);
        triangle.InsertEdge(edge, edgeDst);
        triangles_.emplace_back(triangle);
        triangles_.back().origin_idx = triangles_.size() - 1;
    }



    std::cout << "Load successful (" << triangles_.size() << " triangles)" << std::endl;
    std::cout << "Load successful (" << edges_.size() << " edges)" << std::endl;

    

}

std::size_t Scene::LoadTexture(char const* filename) {
    // Try to lookup the cache
    auto it = loaded_textures_.find(filename);
    if (it != loaded_textures_.cend()) {
        return it->second;
    }

    // Load the texture
    char const* file_extension = strrchr(filename, '.');
    if (file_extension == nullptr) {
        throw std::runtime_error("Invalid texture extension");
    }

    bool success = false;
    Image image;
    if (strcmp(file_extension, ".hdr") == 0) {
        assert(!"Not implemented yet!");
        success = LoadHDR(filename, image);
    }
    else if (strcmp(file_extension, ".jpg") == 0 || strcmp(file_extension, ".tga") == 0 || strcmp(file_extension, ".png") == 0) {
        success = LoadSTB(filename, image);
    }

    if (!success) {
        throw std::runtime_error((std::string("Failed to load file ") + filename).c_str());
    }

    Texture texture;
    texture.width = image.width;
    texture.height = image.height;
    texture.data_start = (std::uint32_t) texture_data_.size();

    std::size_t texture_idx = textures_.size();
    textures_.push_back(std::move(texture));

    texture_data_.insert(texture_data_.end(), image.data.begin(), image.data.end());

    // Cache the texture
    loaded_textures_.emplace(filename, texture_idx);
    return texture_idx;
}

void Scene::CollectEmissiveTriangles() {
    for (auto triangle_idx = 0; triangle_idx < triangles_.size(); ++triangle_idx) {
        auto const& triangle = triangles_[triangle_idx];
        float3 emission = UnpackRGBE(materials_[0].emission);

        if (emission.x + emission.y + emission.z > 0.0f) {
            // The triangle is emissive
            emissive_indices_.push_back(triangle_idx);
        }
    }

    scene_info_.emissive_count = (std::uint32_t) emissive_indices_.size();
}

void Scene::AddPointLight(float3 origin, float3 radiance) {
    Light light = { origin, radiance, LIGHT_TYPE_POINT };
    lights_.push_back(std::move(light));
}

void Scene::AddDirectionalLight(float3 direction, float3 radiance) {
    Light light = { direction.Normalize(), radiance, LIGHT_TYPE_DIRECTIONAL };
    lights_.emplace_back(std::move(light));
}

void Scene::Finalize() {
    CollectEmissiveTriangles();

    //scene_info_.environment_map_index = LoadTexture("textures/studio_small_03_4k.hdr");
    scene_info_.analytic_light_count = (std::uint32_t) lights_.size();

    LoadHDR("assets/ibl/CGSkies_0036_free.hdr", env_image_);
}

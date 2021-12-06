#ifndef RAYTRIANGLEINTERSECTION_H
#define RAYTRIANGLEINTERSECTION_H

#include "TriMesh.h"
#include <iostream>

const float epsilon = 1e-6;

bool checkRayTriangleIntersected(const trimesh::point &ray_start, trimesh::point ray_dir, const trimesh::point &v0, const trimesh::point &v1, const trimesh::point &v2, float &d)
{
    trimesh::normalize(ray_dir);

    trimesh::vec3 v0v1 = v1 - v0;
    trimesh::vec3 v0v2 = v2 - v0;
    trimesh::vec3 tri_norm = trimesh::cross(v0v1, v0v2);
    std::cout<<"ray_start: "<<ray_start<<std::endl;
    std::cout<<"v0: "<<v0<<std::endl;
    std::cout<<"v1: "<<v1<<std::endl;
    std::cout<<"v2: "<<v2<<std::endl;
    trimesh::normalize(tri_norm);
    std::cout<<"tri_norm: "<<tri_norm<<std::endl;
    trimesh::point tmp_p = ray_start - v0;

    // check if ray and plane are parallel ?
    float raydir_dot_trinorm = trimesh::dot(ray_dir, tri_norm);
    if (std::fabs(raydir_dot_trinorm) < epsilon) // almost 0
    {
        return false;   // they are parallel so they don't intersect !
    }

    float tmp_p_dot_trinorm = trimesh::dot(tmp_p, tri_norm);
    d = -1.f * tmp_p_dot_trinorm / raydir_dot_trinorm;

    if (d < 0)  // the triangle is behind
    {
        return false;
    }

    trimesh::point p_hit = d*ray_dir + ray_start;
    std::cout<<"d: "<<d<<std::endl;
    std::cout<<"p_hit: "<<p_hit<<std::endl;

    trimesh::vec3 edge0 = v1 - v0;
    trimesh::vec3 vp0 = p_hit - v0;
    trimesh::vec3 cross0 = trimesh::cross(edge0, vp0);
    std::cout<<"edge0: "<<edge0<<std::endl;
    std::cout<<"vp0: "<<vp0<<std::endl;
    std::cout<<"cross0: "<<cross0<<std::endl;
    if (trimesh::dot(tri_norm, cross0) < -epsilon)
    {
        std::cout<<"dot0: "<<trimesh::dot(tri_norm, cross0)<<std::endl;
        return false;   // P is outside the triangle
    }

    trimesh::vec3 edge1 = v2 - v1;
    trimesh::vec3 vp1 = p_hit - v1;
    trimesh::vec3 cross1 = trimesh::cross(edge1, vp1);
    std::cout<<"edge1: "<<edge1<<std::endl;
    std::cout<<"vp1: "<<vp1<<std::endl;
    std::cout<<"cross1: "<<cross1<<std::endl;
    if (trimesh::dot(tri_norm, cross1) < -epsilon)
    {
        std::cout<<"1: "<<trimesh::dot(tri_norm, cross1)<<std::endl;
        return false;   // P is outside the triangle
    }

    trimesh::vec3 edge2 = v0 - v2;
    trimesh::vec3 vp2 = p_hit - v2;
    trimesh::vec3 cross2 = trimesh::cross(edge2, vp2);
    std::cout<<"edge2: "<<edge2<<std::endl;
    std::cout<<"vp2: "<<vp2<<std::endl;
    std::cout<<"cross2: "<<cross2<<std::endl;
    if (trimesh::dot(tri_norm, cross2) < -epsilon)
    {
        std::cout<<"2: "<<trimesh::dot(tri_norm, cross2)<<std::endl;
        return false;   // P is outside the triangle
    }

    return true;
}

bool checkRayTriangleIntersected2(const trimesh::point &ray_ori, const trimesh::point &ray_dir, const trimesh::point &v0, const trimesh::point &v1, const trimesh::point &v2, float &t)
{
    // The algorithm is based on solving a linear system of equations defined by
    // equality of any point on triangle represented using barycentric coordinates and a ray formula:
    //
    // (1 - u - v) * V0 + u * V1 + v * V2    =   Origin + t * direction
    //
    // The algorithm uses Cramer's rule to solve the equation.

    // Determine vectors for 2 edges sharing Vertex 0
    trimesh::vec3 edge1 = v1 - v0;
    trimesh::vec3 edge2 = v2 - v0;

    trimesh::vec3 h = trimesh::cross(ray_dir, edge2);
    float determinant = trimesh::dot(edge1, h);

    // Negative determinant would indicate back facing triangle
    // and could theoretically be ignored and early out.
    // This implementation ignores this and assumes double-sided triangle.
    // Very small determinant means the ray is almost parallel with the triangle plane.
    if (determinant > -epsilon && determinant < epsilon)
    {
        return false;
    }

    float inverse_determinant = 1.0 / determinant;

    trimesh::vec3 origins_diff_vector = ray_ori - v0;
    float u = trimesh::dot(origins_diff_vector, h)*inverse_determinant;

    // Check the u-barycentric coordinate for validity to save further expensive calculations
    if (u < 0.0 || u > 1.0)
    {
        return false;
    }

    trimesh::vec3 q = trimesh::cross(origins_diff_vector, edge1);
    float v = inverse_determinant * trimesh::dot(ray_dir, q);

    // Check the v-barycentric coordinate for validity to save further expensive calculations
    if (v < 0.0 || u + v > 1.0)
    {
        return false;
    }

    // At this stage we can compute t to find out where the intersection point is on the line
    t = inverse_determinant * trimesh::dot(edge2, q);

    if (t > epsilon)
    {
//            intersection_point = ray_ori + ray_dir * t;
        return true;
    }
    else
    {
        // This means that there is a line intersection but not a ray intersection
        return false;
    }
}

#endif // RAYTRIANGLEINTERSECTION_H

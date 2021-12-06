#ifndef BASEPROCESSOR_H
#define BASEPROCESSOR_H

#include "TriMesh.h"

class BaseProcessor
{
public:
    void    subdivideMeshMPS(trimesh::TriMesh *mesh);
    void    subdivideMeshPartMPS(trimesh::TriMesh *mesh, const std::vector<int> &face_idx, bool with_color=true);
    std::vector<int>    subdivideMeshPartMPS(trimesh::TriMesh *mesh, std::vector<int> &face_idx, std::vector<bool> &b_face_sub, int sub_iter=1, bool with_color=true);    // return new face index of subdivided face_idx
    std::vector<int>    subdivideMeshPartCS(trimesh::TriMesh *mesh, const std::vector<int> &face_idx, std::vector<bool> &b_face_sub);     // center face subdivide

public:
    void    doLaplacianSmooth(trimesh::TriMesh *mesh, const std::vector<int> &vidx, const float &lambda=0.85, const int &n_iter=3);
protected:
    bool    calcCotangentWeights(const trimesh::point &p, const std::vector<trimesh::point> &p_nei, std::vector<float> &w);

public:
    BaseProcessor();
};

#endif // BASEPROCESSOR_H

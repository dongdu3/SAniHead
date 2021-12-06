#include "BaseProcessor.h"
#include <assert.h>

BaseProcessor::BaseProcessor()
{

}

bool BaseProcessor::calcCotangentWeights(const trimesh::point &p, const std::vector<trimesh::point> &p_nei, std::vector<float> &w)
{
    int n = p_nei.size();
    if (n < 3)
    {
        return false;
    }

    w.resize(n, 0);

    // for the first neighbor vertex
    float sum_w = 0;

    trimesh::vec3 a1 = p_nei[0] - p_nei[n-1];
    trimesh::vec3 b1 = p - p_nei[n-1];
    trimesh::vec3 a2 = p - p_nei[1];
    trimesh::vec3 b2 = p_nei[0] - p_nei[1];
    w[0] = a1.dot(b1)/trimesh::len(a1.cross(b1)) + a2.dot(b2)/trimesh::len(a2.cross(b2));

    sum_w += w[0];

    // for the last neighbor vertex
    a1 = p_nei[n-1] - p_nei[n-2];
    b1 = p - p_nei[n-2];
    a2 = p - p_nei[0];
    b2 = p_nei[n-1] - p_nei[0];
    w[n-1] = a1.dot(b1)/trimesh::len(a1.cross(b1)) + a2.dot(b2)/trimesh::len(a2.cross(b2));

    sum_w += w[n-1];

    // for the middle neighbor vertices
    for (int i=1; i<n-1; ++i)
    {
        a1 = p_nei[i] - p_nei[i-1];
        b1 = p - p_nei[i-1];
        a2 = p - p_nei[i+1];
        b2 = p_nei[i] - p_nei[i+1];
        w[i] = a1.dot(b1)/trimesh::len(a1.cross(b1)) + a2.dot(b2)/trimesh::len(a2.cross(b2));

        sum_w += w[i];
    }

    if (sum_w>1e-4 && sum_w<1e5)
    {
        for (int i=0; i<n; ++i)
        {
            w[i] /= sum_w;
        }

        return true;
    }
    else
    {
        w.clear();
        w.resize(n, 1./n);

        return false;
    }
}

void BaseProcessor::doLaplacianSmooth(trimesh::TriMesh *mesh, const std::vector<int> &vidx, const float &lambda, const int &n_iter)
{
    const int n = vidx.size();
    if (n > 0)
    {
        mesh->need_neighbors();
        std::vector<trimesh::point> p_new(n, trimesh::point(0, 0, 0));

        for (int k=0; k<n_iter; ++k)
        {
            for (int i=0; i<n; ++i)
            {
                int id = vidx[i];
                const std::vector<int> &v_nei_id = mesh->neighbors[id];
                std::vector<trimesh::point> p_nei;
                std::vector<float> w;
                for (int j=0; j<v_nei_id.size(); ++j)
                {
                    p_nei.push_back(mesh->vertices[v_nei_id[j]]);
                }
                calcCotangentWeights(mesh->vertices[id], p_nei, w);

                trimesh::point p(0, 0, 0);
                for (int j=0; j<v_nei_id.size(); ++j)
                {
                    p += mesh->vertices[v_nei_id[j]]*w[j];
                }

                p_new[i] = mesh->vertices[id] + lambda*(p-mesh->vertices[id]);
            }

            for (int i=0; i<n; ++i)
            {
                mesh->vertices[vidx[i]] = p_new[i];
            }
        }
    }
}

void BaseProcessor::subdivideMeshMPS(trimesh::TriMesh *mesh)   // middle point subdivision (whole mesh)
{
    assert(mesh && !mesh->vertices.empty());

    int nv = mesh->vertices.size();
    int nv_new = nv;
    int nf = mesh->faces.size();
    std::vector<trimesh::TriMesh::Face> faces = mesh->faces;
    std::vector<std::vector<int>> edge_v_id(nv, std::vector<int>(nv, -1));

    bool color_flag = false;
    if (mesh->colors.size() > 0)
    {
        color_flag = true;
    }

    // add vertices
    for (int i=0; i<nf; ++i)
    {
        const trimesh::TriMesh::Face &f = faces[i];
        for (int j=0; j<3; ++j)
        {
            int id1 = f[j];
            int id2 = f[(j+1)%3];
            if (edge_v_id[id1][id2]<0 && edge_v_id[id2][id1]<0)
            {
                trimesh::point p = (mesh->vertices[id1] + mesh->vertices[id2])/2.f;
                mesh->vertices.push_back(p);
                if (color_flag)
                {
                    mesh->colors.push_back((mesh->colors[id1]+mesh->colors[id2])/2.f);
                }

                edge_v_id[id1][id2] = nv_new;
                edge_v_id[id2][id1] = nv_new;
                nv_new++;
            }
        }
    }

    // add faces
    mesh->faces.clear();
    for (int i=0; i<nf; ++i)
    {
        const trimesh::TriMesh::Face &f = faces[i];
        int idx[6] = {f[0], f[1], f[2], edge_v_id[f[0]][f[1]], edge_v_id[f[1]][f[2]], edge_v_id[f[2]][f[0]]};
        mesh->faces.push_back(trimesh::TriMesh::Face(idx[0], idx[3], idx[5]));
        mesh->faces.push_back(trimesh::TriMesh::Face(idx[3], idx[1], idx[4]));
        mesh->faces.push_back(trimesh::TriMesh::Face(idx[5], idx[4], idx[2]));
        mesh->faces.push_back(trimesh::TriMesh::Face(idx[3], idx[4], idx[5]));
    }

    mesh->clear_normals();
    mesh->clear_neighbors();

//    std::cout<<"middle point subdivision done."<<std::endl;
//    std::cout<<"new mesh nv: "<<mesh->vertices.size()<<std::endl;
//    std::cout<<"new mesh nf: "<<mesh->faces.size()<<std::endl;
}

void BaseProcessor::subdivideMeshPartMPS(trimesh::TriMesh *mesh, const std::vector<int> &face_idx, bool with_color)
{
    assert(mesh && !mesh->vertices.empty());

    if (face_idx.size() > 0)
    {
        // check face information for subdivition
        int nv = mesh->vertices.size();
        std::vector<trimesh::TriMesh::Face> faces = mesh->faces;
        std::vector<std::vector<int>> edge_opposite_v_id(nv, std::vector<int>(nv, -1));
        for (int i=0; i<face_idx.size(); ++i)
        {
            const trimesh::TriMesh::Face &f = faces[face_idx[i]];
            edge_opposite_v_id[f[0]][f[1]] = f[2];
            edge_opposite_v_id[f[1]][f[2]] = f[0];
            edge_opposite_v_id[f[2]][f[0]] = f[1];
        }

        // pick out the faces to subdivide
        std::vector<bool> b_f_remain(faces.size(), true);
        std::vector<int> sub_face_id_ori;
        for (int i=0; i<face_idx.size(); ++i)
        {
            const int fid = face_idx[i];
            bool do_sub = false;
            const trimesh::TriMesh::Face &f = faces[fid];
            for (int j=0; j<3; ++j)
            {
                if (edge_opposite_v_id[f[j]][f[(j+1)%3]]>=0 && edge_opposite_v_id[f[(j+1)%3]][f[j]]>=0)
                {
                    do_sub = true;
                    break;
                }
            }

            if (do_sub)
            {
                sub_face_id_ori.push_back(fid);
                b_f_remain[fid] = false;
            }
        }

        // remain original faces
        mesh->faces.clear();
        int nf_new = 0;
        for (int i=0; i<faces.size(); ++i)
        {
            if (b_f_remain[i])
            {
                mesh->faces.push_back(faces[i]);
                nf_new++;
            }
        }

        // add vertices and faces
        int nv_new = nv;
        std::vector<std::vector<int>> edge_v_id(nv, std::vector<int>(nv, -1));

        // add vertices and faces
        for (int i=0; i<sub_face_id_ori.size(); ++i)
        {
            const int fid = sub_face_id_ori[i];
            const trimesh::TriMesh::Face &f = faces[fid];
            bool b_edge_sub[3] = {false, false, false};
            int n_edge_sub = 0;
            // add vertices
            for (int j=0; j<3; ++j)
            {
                int id1 = f[j];
                int id2 = f[(j+1)%3];
                if (edge_opposite_v_id[id1][id2]>=0 && edge_opposite_v_id[id2][id1]>=0)
                {
                    b_edge_sub[j] = true;
                    n_edge_sub++;

                    if (edge_v_id[id1][id2]<0 && edge_v_id[id2][id1]<0)
                    {
                        trimesh::point p = (mesh->vertices[id1] + mesh->vertices[id2])/2.f;
                        mesh->vertices.push_back(p);
                        if (with_color)
                        {
                            mesh->colors.push_back((mesh->colors[id1]+mesh->colors[id2])/2.f);
                        }

                        edge_v_id[id1][id2] = nv_new;
                        edge_v_id[id2][id1] = nv_new;
                        nv_new++;
                    }
                }
            }

            // add faces
            if (n_edge_sub==1)  // add two faces
            {
                for (int j=0; j<3; ++j)
                {
                    if (b_edge_sub[j])
                    {
                        const int id1 = f[j];
                        const int id2 = f[(j+1)%3];
                        const int id3 = f[(j+2)%3];
                        const int id4 = edge_v_id[id1][id2];

                        mesh->faces.push_back(trimesh::TriMesh::Face(id1, id4, id3));
                        nf_new++;

                        mesh->faces.push_back(trimesh::TriMesh::Face(id4, id2, id3));
                        nf_new++;

                        break;
                    }
                }
            }
            else if (n_edge_sub == 2)   // add three faces
            {
                for (int j=0; j<3; ++j)
                {
                    if (b_edge_sub[j] && b_edge_sub[(j+1)%3])
                    {
                        const int id1 = f[j];
                        const int id2 = f[(j+1)%3];
                        const int id3 = f[(j+2)%3];
                        const int id4 = edge_v_id[id1][id2];
                        const int id5 = edge_v_id[id2][id3];

                        mesh->faces.push_back(trimesh::TriMesh::Face(id1, id4, id3));
                        nf_new++;

                        mesh->faces.push_back(trimesh::TriMesh::Face(id4, id5, id3));
                        nf_new++;

                        mesh->faces.push_back(trimesh::TriMesh::Face(id4, id2, id5));
                        nf_new++;
                    }
                }
            }
            else if (n_edge_sub == 3)   // add four faces
            {
                int idx[6] = {f[0], f[1], f[2], edge_v_id[f[0]][f[1]], edge_v_id[f[1]][f[2]], edge_v_id[f[2]][f[0]]};
                mesh->faces.push_back(trimesh::TriMesh::Face(idx[0], idx[3], idx[5]));
                mesh->faces.push_back(trimesh::TriMesh::Face(idx[3], idx[1], idx[4]));
                mesh->faces.push_back(trimesh::TriMesh::Face(idx[5], idx[4], idx[2]));
                mesh->faces.push_back(trimesh::TriMesh::Face(idx[3], idx[4], idx[5]));

                for (int j=0; j<4; ++j)
                {
                    nf_new++;
                }
            }
        }

//        std::cout<<"middle point subdivision done.";
//        std::cout<<"new mesh nv: "<<mesh->vertices.size()<<std::endl;
//        std::cout<<"new mesh nf: "<<mesh->faces.size()<<std::endl;

        mesh->clear_normals();
        mesh->clear_neighbors();
    }
}

std::vector<int> BaseProcessor::subdivideMeshPartMPS(trimesh::TriMesh *mesh, std::vector<int> &face_idx, std::vector<bool> &b_face_sub, int sub_iter, bool with_color)   // middle point subdivision (partial mesh), return new face index of subdivided face_idx
{
    assert(mesh && !mesh->vertices.empty());

    if (face_idx.size() > 0 && sub_iter > 0)
    {
        std::vector<int> face_select_idx;
        for (int k=0; k<sub_iter; ++k)
        {
            // check face information for subdivition
            int nv = mesh->vertices.size();
            std::vector<trimesh::TriMesh::Face> faces = mesh->faces;
            std::vector<bool> b_f_select(faces.size(), false);
            std::vector<std::vector<int>> edge_opposite_v_id(nv, std::vector<int>(nv, -1));
            for (int i=0; i<face_idx.size(); ++i)
            {
                const int fid = face_idx[i];
                b_f_select[fid] = true;

                if (!b_face_sub[fid])    // face hasn't been subdivided
                {
                    const trimesh::TriMesh::Face &f = faces[fid];
                    edge_opposite_v_id[f[0]][f[1]] = f[2];
                    edge_opposite_v_id[f[1]][f[2]] = f[0];
                    edge_opposite_v_id[f[2]][f[0]] = f[1];
                }
            }

            // pick out the faces to subdivide
            std::vector<bool> b_f_remain(faces.size(), true);
            std::vector<int> sub_face_id_ori;
            for (int i=0; i<face_idx.size(); ++i)
            {
                const int fid = face_idx[i];
                if (!b_face_sub[fid])
                {
                    bool do_sub = false;
                    const trimesh::TriMesh::Face &f = faces[fid];
                    for (int j=0; j<3; ++j)
                    {
                        if (edge_opposite_v_id[f[j]][f[(j+1)%3]]>=0 && edge_opposite_v_id[f[(j+1)%3]][f[j]]>=0)
                        {
                            do_sub = true;
                            break;
                        }
                    }

                    if (do_sub)
                    {
                        sub_face_id_ori.push_back(fid);
                        b_f_remain[fid] = false;
                    }
                }
            }

            // remain original faces
            std::vector<bool> b_face_sub_bk = b_face_sub;
            b_face_sub.clear();
            mesh->faces.clear();
            int nf_new = 0;
            face_select_idx.clear();
            for (int i=0; i<faces.size(); ++i)
            {
                if (b_f_remain[i])
                {
                    mesh->faces.push_back(faces[i]);
                    b_face_sub.push_back(b_face_sub_bk[i]);

                    if (b_f_select[i])
                    {
                        face_select_idx.push_back(nf_new);
                    }

                    nf_new++;
                }
            }

            // add vertices and faces
            int nv_new = nv;
            std::vector<std::vector<int>> edge_v_id(nv, std::vector<int>(nv, -1));

            // add vertices and faces
            for (int i=0; i<sub_face_id_ori.size(); ++i)
            {
                const int fid = sub_face_id_ori[i];
                const trimesh::TriMesh::Face &f = faces[fid];
                bool b_edge_sub[3] = {false, false, false};
                int n_edge_sub = 0;
                // add vertices
                for (int j=0; j<3; ++j)
                {
                    int id1 = f[j];
                    int id2 = f[(j+1)%3];
                    if (edge_opposite_v_id[id1][id2]>=0 && edge_opposite_v_id[id2][id1]>=0)
                    {
                        b_edge_sub[j] = true;
                        n_edge_sub++;

                        if (edge_v_id[id1][id2]<0 && edge_v_id[id2][id1]<0)
                        {
                            trimesh::point p = (mesh->vertices[id1] + mesh->vertices[id2])/2.f;
                            mesh->vertices.push_back(p);
                            if (with_color)
                            {
                                mesh->colors.push_back((mesh->colors[id1]+mesh->colors[id2])/2.f);
                            }

                            edge_v_id[id1][id2] = nv_new;
                            edge_v_id[id2][id1] = nv_new;
                            nv_new++;
                        }
                    }
                }

                // add faces
                if (n_edge_sub==1)  // add two faces
                {
                    for (int j=0; j<3; ++j)
                    {
                        if (b_edge_sub[j])
                        {
                            const int id1 = f[j];
                            const int id2 = f[(j+1)%3];
                            const int id3 = f[(j+2)%3];
                            const int id4 = edge_v_id[id1][id2];

                            mesh->faces.push_back(trimesh::TriMesh::Face(id1, id4, id3));
                            if (k < sub_iter-1)
                            {
                                b_face_sub.push_back(false);
                            }
                            else
                            {
                                b_face_sub.push_back(true);
                            }
                            face_select_idx.push_back(nf_new);
                            nf_new++;

                            mesh->faces.push_back(trimesh::TriMesh::Face(id4, id2, id3));
                            if (k < sub_iter-1)
                            {
                                b_face_sub.push_back(false);
                            }
                            else
                            {
                                b_face_sub.push_back(true);
                            }
                            face_select_idx.push_back(nf_new);
                            nf_new++;

                            break;
                        }
                    }
                }
                else if (n_edge_sub == 2)   // add three faces
                {
                    for (int j=0; j<3; ++j)
                    {
                        if (b_edge_sub[j] && b_edge_sub[(j+1)%3])
                        {
                            const int id1 = f[j];
                            const int id2 = f[(j+1)%3];
                            const int id3 = f[(j+2)%3];
                            const int id4 = edge_v_id[id1][id2];
                            const int id5 = edge_v_id[id2][id3];

                            mesh->faces.push_back(trimesh::TriMesh::Face(id1, id4, id3));
                            if (k < sub_iter-1)
                            {
                                b_face_sub.push_back(false);
                            }
                            else
                            {
                                b_face_sub.push_back(true);
                            }
                            face_select_idx.push_back(nf_new);
                            nf_new++;

                            mesh->faces.push_back(trimesh::TriMesh::Face(id4, id5, id3));
                            if (k < sub_iter-1)
                            {
                                b_face_sub.push_back(false);
                            }
                            else
                            {
                                b_face_sub.push_back(true);
                            }
                            face_select_idx.push_back(nf_new);
                            nf_new++;

                            mesh->faces.push_back(trimesh::TriMesh::Face(id4, id2, id5));
                            if (k < sub_iter-1)
                            {
                                b_face_sub.push_back(false);
                            }
                            else
                            {
                                b_face_sub.push_back(true);
                            }
                            face_select_idx.push_back(nf_new);
                            nf_new++;
                        }
                    }
                }
                else if (n_edge_sub == 3)   // add four faces
                {
                    int idx[6] = {f[0], f[1], f[2], edge_v_id[f[0]][f[1]], edge_v_id[f[1]][f[2]], edge_v_id[f[2]][f[0]]};
                    mesh->faces.push_back(trimesh::TriMesh::Face(idx[0], idx[3], idx[5]));
                    mesh->faces.push_back(trimesh::TriMesh::Face(idx[3], idx[1], idx[4]));
                    mesh->faces.push_back(trimesh::TriMesh::Face(idx[5], idx[4], idx[2]));
                    mesh->faces.push_back(trimesh::TriMesh::Face(idx[3], idx[4], idx[5]));

                    for (int j=0; j<4; ++j)
                    {
                        if (k < sub_iter-1)
                        {
                            b_face_sub.push_back(false);
                        }
                        else
                        {
                            b_face_sub.push_back(true);
                        }
                        face_select_idx.push_back(nf_new);
                        nf_new++;
                    }
                }
            }

            face_idx = face_select_idx;
        }

//        std::cout<<"middle point subdivision done.";
//        std::cout<<"new mesh nv: "<<mesh->vertices.size()<<std::endl;
//        std::cout<<"new mesh nf: "<<mesh->faces.size()<<std::endl;

        mesh->clear_normals();
        mesh->clear_neighbors();

        return face_select_idx;
    }
}

std::vector<int> BaseProcessor::subdivideMeshPartCS(trimesh::TriMesh *mesh, const std::vector<int> &face_idx, std::vector<bool> &b_face_sub)   // subdivision (partial mesh), return new face index of subdivided face_idx
{
    assert(mesh && !mesh->vertices.empty());

    if (face_idx.size() > 0)
    {
        int nv = mesh->vertices.size();
        int nv_sub = nv;
        std::vector<trimesh::TriMesh::Face> faces = mesh->faces;

        bool color_flag = false;
        if (mesh->colors.size() > 0)
        {
            color_flag = true;
        }

        // add vertices
        std::vector<bool> b_f_remain(faces.size(), true);
        std::vector<bool> b_f_select(faces.size(), false);
        std::vector<int> face_sub_vertex_id(faces.size(), -1);
        std::vector<int> sub_face_idx_ori;
        for (int i=0; i<face_idx.size(); ++i)
        {
            const int fid = face_idx[i];
            b_f_select[fid] = true;

            if (!b_face_sub[fid])    // face hasn't been subdivided
            {
                b_f_remain[fid] = false;
                sub_face_idx_ori.push_back(fid);

                const trimesh::TriMesh::Face &f = faces[fid];
                mesh->vertices.push_back((mesh->vertices[f[0]] + mesh->vertices[f[1]] + mesh->vertices[f[2]])/3.f);
                if (color_flag)
                {
                    mesh->colors.push_back((mesh->colors[f[0]] + mesh->colors[f[1]] + mesh->colors[f[2]])/3.f);
                }

                face_sub_vertex_id[fid] = nv_sub;
                nv_sub++;
            }
        }

        // remain original faces
        std::vector<bool> b_face_sub_bk = b_face_sub;
        b_face_sub.clear();
        mesh->faces.clear();
        int tmp_id = 0;
        std::vector<int> face_idx_select;
        for (int i=0; i<faces.size(); ++i)
        {
            if (b_f_remain[i])
            {
                mesh->faces.push_back(faces[i]);
                b_face_sub.push_back(b_face_sub_bk[i]);

                if (b_f_select[i])
                {
                    face_idx_select.push_back(tmp_id);
                }

                tmp_id++;
            }
        }

        // add new faces
        for (int i=0; i<sub_face_idx_ori.size(); ++i)
        {
            const trimesh::TriMesh::Face &f = faces[sub_face_idx_ori[i]];
            int idx[4] = {f[0], f[1], f[2], face_sub_vertex_id[sub_face_idx_ori[i]]};
            mesh->faces.push_back(trimesh::TriMesh::Face(idx[0], idx[1], idx[3]));
            mesh->faces.push_back(trimesh::TriMesh::Face(idx[1], idx[2], idx[3]));
            mesh->faces.push_back(trimesh::TriMesh::Face(idx[2], idx[0], idx[3]));
            for (int j=0; j<3; ++j)
            {
                b_face_sub.push_back(true);
                face_idx_select.push_back(tmp_id);

                tmp_id++;
            }
        }

//        std::cout<<"middle point subdivision done.";
//        std::cout<<"new mesh nv: "<<mesh->vertices.size()<<std::endl;
//        std::cout<<"new mesh nf: "<<mesh->faces.size()<<std::endl;

        mesh->clear_normals();
        mesh->clear_neighbors();

        return face_idx_select;
    }
}

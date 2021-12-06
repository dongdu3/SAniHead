#ifndef DIJKSTRASHORTESTPATH_H
#define DIJKSTRASHORTESTPATH_H

#include "TriMesh.h"
#include "Vec.h"
#include <vector>

class CDijkstraShortestPath
{
public:
    CDijkstraShortestPath();
    ~CDijkstraShortestPath();

public:
    float	calculateRelativeDistance(int v_index1,int v_index2, const trimesh::TriMesh *mesh, bool with_feat_metric=false);
    void	getDijkstraPath(std::vector<int> &route_index, const int v_index1, const int v_index2, const trimesh::TriMesh *mesh, bool with_feat_metric=false);

private:
	
};

#endif // DIJKSTRASHORTESTPATH_H

#include "DijkstraShortestPath.h"
#include <assert.h>

#define MAX_DISTANCE 10e10

CDijkstraShortestPath::CDijkstraShortestPath()
{

}

CDijkstraShortestPath::~CDijkstraShortestPath()
{

}

float CDijkstraShortestPath::calculateRelativeDistance(int v_index1,int v_index2, const trimesh::TriMesh *mesh, bool with_feat_metric)
{
	if ( v_index1 == v_index2 )
	{
		return 0;
	}
	else
	{
		for ( int i=0; i<mesh->neighbors[v_index1].size(); ++i )
		{
			if ( v_index2 == mesh->neighbors[v_index1][i] )
			{
                if (with_feat_metric)
                {
//                    return len(mesh->colors[v_index1]-mesh->colors[v_index2]);
                    return 1.8f - len(mesh->colors[v_index2]);
                }
                else
                {
                    return len(mesh->vertices[v_index1]-mesh->vertices[v_index2]);
                }
			}
		}
	}
	
	return MAX_DISTANCE;
}

void CDijkstraShortestPath::getDijkstraPath(std::vector<int> &route_index, const int v_index1, const int v_index2, const trimesh::TriMesh *mesh, bool with_feat_metric)
{
    assert(mesh);

    if (with_feat_metric)
    {
        assert(mesh->vertices.size()==mesh->colors.size());
    }

	route_index.clear();

    int num_of_vertex = mesh->vertices.size();  // the number of nodes
    std::vector<float> dist;    // distance to the source node
    std::vector<int> pre_vertex;  // the pre-node of every node
    std::vector<bool> is_in;      // if in the set of S

	dist.clear();
	pre_vertex.clear();
	is_in.clear();
	dist.resize(num_of_vertex);
	pre_vertex.resize(num_of_vertex);
    is_in.assign(num_of_vertex, false);     // initilization, no nodes in the S

    // initilization
	for(int k =0; k < num_of_vertex; k++)  
	{  
        dist[k] = calculateRelativeDistance(v_index1, k, mesh, with_feat_metric);
		pre_vertex[k] = v_index1;
	} 

    is_in[v_index1] = true;
    int current_point_index = v_index1;
    float current_least_dist = 0;

	while (!is_in[v_index2])
	{
		float least_dist = MAX_DISTANCE;
		for (int k=0; k<num_of_vertex; k++)
		{
			if (!is_in[k])
			{
				if (dist[k]<least_dist)
				{
					least_dist = dist[k];
					current_point_index = k;
				}
			}
		}
		current_least_dist = least_dist;
		is_in[current_point_index] = true;
		for (int k=0; k<num_of_vertex; k++)
		{
			if (!is_in[k])
			{
                float new_dist = calculateRelativeDistance(current_point_index, k, mesh, with_feat_metric)+current_least_dist;
				if (new_dist<dist[k])
				{
					dist[k] = new_dist;
					pre_vertex[k] = current_point_index;
				}
			}
		}
	}
	route_index.push_back(v_index2);

	int m = pre_vertex[v_index2];
	do 
	{
		route_index.push_back(m);
		m = pre_vertex[m];
	} while ( m!=v_index1 );
	route_index.push_back(v_index1);

	std::vector<int> temp = route_index;
	for (int k=0; k<temp.size(); k++)
	{
		route_index[k] = temp[temp.size()-k-1];
	}
}

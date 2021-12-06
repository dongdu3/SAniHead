#pragma once

#include "Vec.h"

struct POINTF {float x; float y;};


class CLineSegmentIntersection
{
public:
	CLineSegmentIntersection(void);
	virtual ~CLineSegmentIntersection(void);

public:
	void	swapTwoPoint(trimesh::vec2 &p1, trimesh::vec2 &p2);
	int		intersectTwoLineSegment(trimesh::vec2 &intersect_p, trimesh::vec2 &line1_p1, trimesh::vec2 &line1_p2, trimesh::vec2 &line2_p1, trimesh::vec2 &line2_p2);
};


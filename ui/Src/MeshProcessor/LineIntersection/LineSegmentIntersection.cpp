#include "LineSegmentIntersection.h"
#include <iostream>
#include <cmath>

/*global function*/

//判断两个数是否相等;
bool Equal(float f1, float f2) 
{
	return (abs(f1 - f2) < 10e-3f);
}

//判断两点是否相等;
bool	operator == (const trimesh::vec2 &p1, const trimesh::vec2 &p2) 
{
	return (Equal(p1[0], p2[0]) && Equal(p1[1], p2[1]));
}
//比较两点坐标大小，先比较x坐标，若相同则比较y坐标;
bool	operator > (const trimesh::vec2 &p1, const trimesh::vec2 &p2) 
{
	return ( p1[0] > p2[0] || (Equal(p1[0], p2[0]) && p1[1] > p2[1]));
}
//计算两向量外积;
float	operator ^ (const trimesh::vec2 &p1, const trimesh::vec2 &p2) 
{
	return (p1[0] * p2[1] - p1[1] * p2[0]);
}

CLineSegmentIntersection::CLineSegmentIntersection(void)
{
}


CLineSegmentIntersection::~CLineSegmentIntersection(void)
{
}

void CLineSegmentIntersection::swapTwoPoint(trimesh::vec2 &p1, trimesh::vec2 &p2)
{
	trimesh::vec2 temp_p = p1;
	p1 = p2;
	p2 = temp_p;
}

//判定两线段位置关系，并求出交点(如果存在)。返回值列举如下：;
//[有重合] 完全重合(6)，1个端点重合且共线(5)，部分重合(4);
//[无重合] 两端点相交(3)，交于线上(2)，正交(1)，无交(0)，参数错误(-1);
int CLineSegmentIntersection::intersectTwoLineSegment(trimesh::vec2 &intersect_p, trimesh::vec2 &line1_p1, trimesh::vec2 &line1_p2, trimesh::vec2 &line2_p1, trimesh::vec2 &line2_p2)
{	
	//保证参数line1_p1!=line1_p2，line2_p1!=line2_p2;
	if ( line1_p1 == line1_p2 || line2_p1 == line2_p2) {
		return -1; //返回-1代表至少有一条线段首尾重合，不能构成线段;
	}
	//为方便运算，保证各线段的起点在前，终点在后;
	if (line1_p1 > line1_p2) {
		swapTwoPoint(line1_p1, line1_p2);
	}
	if (line2_p1 > line2_p2) {
		swapTwoPoint(line2_p1, line2_p2);
	}
	//判定两线段是否完全重合;
	if (line1_p1 == line2_p1 && line1_p2 == line2_p2) {
		return 6;
	}
	//求出两线段构成的向量;
	trimesh::vec2 v1 = trimesh::vec2(line1_p2[0] - line1_p1[0], line1_p2[1] - line1_p1[1]);
	trimesh::vec2 v2 = trimesh::vec2(line2_p2[0] - line2_p1[0], line2_p2[1] - line2_p1[1]);
	//求两向量外积，平行时外积为0;
	float Corss = v1 ^ v2;
	//如果起点重合;
	if (line1_p1 == line2_p1) {
		intersect_p = line1_p1;
		//起点重合且共线(平行)返回5；不平行则交于端点，返回3;
		return (Equal(Corss, 0) ? 5 : 3);
	}
	//如果终点重合;
	if (line1_p2 == line2_p2) {
		intersect_p = line1_p2;
		//终点重合且共线(平行)返回5；不平行则交于端点，返回3;
		return (Equal(Corss, 0) ? 5 : 3);
	}
	//如果两线端首尾相连;
	if (line1_p1 == line2_p2) {
		intersect_p = line1_p1;
		return 3;
	}
	if (line1_p2 == line2_p1) {
		intersect_p = line1_p2;
		return 3;
	}//经过以上判断，首尾点相重的情况都被排除了;
	//将线段按起点坐标排序。若线段1的起点较大，则将两线段交换;
	if (line1_p1 > line2_p1) {
		swapTwoPoint(line1_p1, line2_p1);
		swapTwoPoint(line1_p2, line2_p2);
		//更新原先计算的向量及其外积;
		swapTwoPoint(v1, v2);
		Corss = v1 ^ v2;
	}
	//处理两线段平行的情况;
	if (Equal(Corss, 0)) {
		//做向量v1(line1_p1, line1_p2)和vs(line1_p1,line2_p1)的外积，判定是否共线;
		trimesh::vec2 vs = trimesh::vec2(line2_p1[0] - line1_p1[0], line2_p1[1] - line1_p1[1]);
		//外积为0则两平行线段共线，下面判定是否有重合部分;
		if (Equal(v1 ^ vs, 0)) {
			//前一条线的终点大于后一条线的起点，则判定存在重合;
			if (line1_p2 > line2_p1) {
				intersect_p = line2_p1;
				return 4; //返回值4代表线段部分重合;
			}
		}//若三点不共线，则这两条平行线段必不共线。;
		//不共线或共线但无重合的平行线均无交点;
		return 0;
	} //以下为不平行的情况，先进行快速排斥试验;
	//x坐标已有序，可直接比较。y坐标要先求两线段的最大和最小值;
	float ymax1 = line1_p1[1], ymin1 = line1_p2[1], ymax2 = line2_p1[1], ymin2 = line2_p2[1];
	if (ymax1 < ymin1) {
		std::swap(ymax1, ymin1);
	}
	if (ymax2 < ymin2) {
		std::swap(ymax2, ymin2);
	}
	//如果以两线段为对角线的矩形不相交，则无交点;
	if (line1_p1[0] > line2_p2[0] || line1_p2[0] < line2_p1[0] || ymax1 < ymin2 || ymin1 > ymax2) {
		return 0;
	}//下面进行跨立试验;
	trimesh::vec2 vs1 = trimesh::vec2(line1_p1[0] - line2_p1[0], line1_p1[1] - line2_p1[1]);
	trimesh::vec2 vs2 = trimesh::vec2(line1_p2[0] - line2_p1[0], line1_p2[1] - line2_p1[1]);
	trimesh::vec2 vt1 = trimesh::vec2(line2_p1[0] - line1_p1[0], line2_p1[1] - line1_p1[1]);
	trimesh::vec2 vt2 = trimesh::vec2(line2_p2[0] - line1_p1[0], line2_p2[1] - line1_p1[1]);
	float s1v2, s2v2, t1v1, t2v1;
	//根据外积结果判定否交于线上;
	if (Equal(s1v2 = vs1 ^ v2, 0) && line2_p2 > line1_p1 && line1_p1 > line2_p1) {
		intersect_p = line1_p1;
		return 2;
	}
	if (Equal(s2v2 = vs2 ^ v2, 0) && line2_p2 > line1_p2 && line1_p2 > line2_p1) {
		intersect_p = line1_p2;
		return 2;
	}
	if (Equal(t1v1 = vt1 ^ v1, 0) && line1_p2 > line2_p1 && line2_p1 > line1_p1) {
		intersect_p = line2_p1;
		return 2;
	}
	if (Equal(t2v1 = vt2 ^ v1, 0) && line1_p2 > line2_p2 && line2_p2 > line1_p1) {
		intersect_p = line2_p2;
		return 2;
	} //未交于线上，则判定是否相交;
	if(s1v2 * s2v2 > 0 || t1v1 * t2v1 > 0) {
		return 0;
	} 

	//计算二阶行列式的两个常数项;
	float ConA = line1_p1[0] * v1[1] - line1_p1[1] * v1[0];
	float ConB = line2_p1[0] * v2[1] - line2_p1[1] * v2[0];
	//计算行列式D1和D2的值，除以系数行列式的值，得到交点坐标;
	intersect_p[0] = (ConB * v1[0] - ConA * v2[0]) / Corss;
	intersect_p[1] = (ConB * v1[1] - ConA * v2[1]) / Corss;
	//正交返回1;
	return 1;
}
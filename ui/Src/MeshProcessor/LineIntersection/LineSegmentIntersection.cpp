#include "LineSegmentIntersection.h"
#include <iostream>
#include <cmath>

/*global function*/

//�ж��������Ƿ����;
bool Equal(float f1, float f2) 
{
	return (abs(f1 - f2) < 10e-3f);
}

//�ж������Ƿ����;
bool	operator == (const trimesh::vec2 &p1, const trimesh::vec2 &p2) 
{
	return (Equal(p1[0], p2[0]) && Equal(p1[1], p2[1]));
}
//�Ƚ����������С���ȱȽ�x���꣬����ͬ��Ƚ�y����;
bool	operator > (const trimesh::vec2 &p1, const trimesh::vec2 &p2) 
{
	return ( p1[0] > p2[0] || (Equal(p1[0], p2[0]) && p1[1] > p2[1]));
}
//�������������;
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

//�ж����߶�λ�ù�ϵ�����������(�������)������ֵ�о����£�;
//[���غ�] ��ȫ�غ�(6)��1���˵��غ��ҹ���(5)�������غ�(4);
//[���غ�] ���˵��ཻ(3)����������(2)������(1)���޽�(0)����������(-1);
int CLineSegmentIntersection::intersectTwoLineSegment(trimesh::vec2 &intersect_p, trimesh::vec2 &line1_p1, trimesh::vec2 &line1_p2, trimesh::vec2 &line2_p1, trimesh::vec2 &line2_p2)
{	
	//��֤����line1_p1!=line1_p2��line2_p1!=line2_p2;
	if ( line1_p1 == line1_p2 || line2_p1 == line2_p2) {
		return -1; //����-1����������һ���߶���β�غϣ����ܹ����߶�;
	}
	//Ϊ�������㣬��֤���߶ε������ǰ���յ��ں�;
	if (line1_p1 > line1_p2) {
		swapTwoPoint(line1_p1, line1_p2);
	}
	if (line2_p1 > line2_p2) {
		swapTwoPoint(line2_p1, line2_p2);
	}
	//�ж����߶��Ƿ���ȫ�غ�;
	if (line1_p1 == line2_p1 && line1_p2 == line2_p2) {
		return 6;
	}
	//������߶ι��ɵ�����;
	trimesh::vec2 v1 = trimesh::vec2(line1_p2[0] - line1_p1[0], line1_p2[1] - line1_p1[1]);
	trimesh::vec2 v2 = trimesh::vec2(line2_p2[0] - line2_p1[0], line2_p2[1] - line2_p1[1]);
	//�������������ƽ��ʱ���Ϊ0;
	float Corss = v1 ^ v2;
	//�������غ�;
	if (line1_p1 == line2_p1) {
		intersect_p = line1_p1;
		//����غ��ҹ���(ƽ��)����5����ƽ�����ڶ˵㣬����3;
		return (Equal(Corss, 0) ? 5 : 3);
	}
	//����յ��غ�;
	if (line1_p2 == line2_p2) {
		intersect_p = line1_p2;
		//�յ��غ��ҹ���(ƽ��)����5����ƽ�����ڶ˵㣬����3;
		return (Equal(Corss, 0) ? 5 : 3);
	}
	//������߶���β����;
	if (line1_p1 == line2_p2) {
		intersect_p = line1_p1;
		return 3;
	}
	if (line1_p2 == line2_p1) {
		intersect_p = line1_p2;
		return 3;
	}//���������жϣ���β�����ص���������ų���;
	//���߶ΰ���������������߶�1�����ϴ������߶ν���;
	if (line1_p1 > line2_p1) {
		swapTwoPoint(line1_p1, line2_p1);
		swapTwoPoint(line1_p2, line2_p2);
		//����ԭ�ȼ���������������;
		swapTwoPoint(v1, v2);
		Corss = v1 ^ v2;
	}
	//�������߶�ƽ�е����;
	if (Equal(Corss, 0)) {
		//������v1(line1_p1, line1_p2)��vs(line1_p1,line2_p1)��������ж��Ƿ���;
		trimesh::vec2 vs = trimesh::vec2(line2_p1[0] - line1_p1[0], line2_p1[1] - line1_p1[1]);
		//���Ϊ0����ƽ���߶ι��ߣ������ж��Ƿ����غϲ���;
		if (Equal(v1 ^ vs, 0)) {
			//ǰһ���ߵ��յ���ں�һ���ߵ���㣬���ж������غ�;
			if (line1_p2 > line2_p1) {
				intersect_p = line2_p1;
				return 4; //����ֵ4�����߶β����غ�;
			}
		}//�����㲻���ߣ���������ƽ���߶αز����ߡ�;
		//�����߻��ߵ����غϵ�ƽ���߾��޽���;
		return 0;
	} //����Ϊ��ƽ�е�������Ƚ��п����ų�����;
	//x���������򣬿�ֱ�ӱȽϡ�y����Ҫ�������߶ε�������Сֵ;
	float ymax1 = line1_p1[1], ymin1 = line1_p2[1], ymax2 = line2_p1[1], ymin2 = line2_p2[1];
	if (ymax1 < ymin1) {
		std::swap(ymax1, ymin1);
	}
	if (ymax2 < ymin2) {
		std::swap(ymax2, ymin2);
	}
	//��������߶�Ϊ�Խ��ߵľ��β��ཻ�����޽���;
	if (line1_p1[0] > line2_p2[0] || line1_p2[0] < line2_p1[0] || ymax1 < ymin2 || ymin1 > ymax2) {
		return 0;
	}//������п�������;
	trimesh::vec2 vs1 = trimesh::vec2(line1_p1[0] - line2_p1[0], line1_p1[1] - line2_p1[1]);
	trimesh::vec2 vs2 = trimesh::vec2(line1_p2[0] - line2_p1[0], line1_p2[1] - line2_p1[1]);
	trimesh::vec2 vt1 = trimesh::vec2(line2_p1[0] - line1_p1[0], line2_p1[1] - line1_p1[1]);
	trimesh::vec2 vt2 = trimesh::vec2(line2_p2[0] - line1_p1[0], line2_p2[1] - line1_p1[1]);
	float s1v2, s2v2, t1v1, t2v1;
	//�����������ж���������;
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
	} //δ�������ϣ����ж��Ƿ��ཻ;
	if(s1v2 * s2v2 > 0 || t1v1 * t2v1 > 0) {
		return 0;
	} 

	//�����������ʽ������������;
	float ConA = line1_p1[0] * v1[1] - line1_p1[1] * v1[0];
	float ConB = line2_p1[0] * v2[1] - line2_p1[1] * v2[0];
	//��������ʽD1��D2��ֵ������ϵ������ʽ��ֵ���õ���������;
	intersect_p[0] = (ConB * v1[0] - ConA * v2[0]) / Corss;
	intersect_p[1] = (ConB * v1[1] - ConA * v2[1]) / Corss;
	//��������1;
	return 1;
}
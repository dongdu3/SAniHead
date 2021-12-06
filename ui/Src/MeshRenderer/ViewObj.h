#ifndef VIEWOBJ_H_
#define VIEWOBJ_H_
#include "GL/glew.h"
#include <stdlib.h>

class ViewObj
{
public:
	ViewObj();
	~ViewObj();
	void updateVBO(GLenum draw_mode = GL_STATIC_DRAW, GLenum Render_mode = GL_POINTS,const unsigned int &nvertices = 0, float* vertices = NULL, float* normals = NULL,
		float *colors = NULL, float *tex_coords = NULL,
		unsigned char *img = NULL, unsigned int iwidth= 0, unsigned int iheight=0);
	void renderVBO();
	void releaseVBO();
protected:
	void releaseVBObuffer(unsigned int& buffer);
	void initVBO();
private:
	bool has_normal_, has_color_, has_vertex_, has_tex_;
	bool has_img_;
	unsigned int vbo_buffer_vertices_;
	unsigned int vbo_buffer_normals_;
	unsigned int vbo_buffer_colors_;
	unsigned int vbo_buffer_tex_;
	GLuint tex_name_;
	unsigned int num_vertices_;
	GLenum render_mode_;
};

#endif

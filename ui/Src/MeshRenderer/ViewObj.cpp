#include "ViewObj.h"
#include <iostream>

ViewObj::ViewObj()
{
	initVBO();
}

ViewObj::~ViewObj()
{
	releaseVBO();
}

void ViewObj::releaseVBObuffer(unsigned int& buffer)
{
	if (buffer == 0) return;
	glDeleteBuffers(1, &buffer);
	buffer = 0;
}

void ViewObj::initVBO()
{
	vbo_buffer_colors_ = 0;
	vbo_buffer_normals_ = 0;
	vbo_buffer_tex_ = 0;
	vbo_buffer_vertices_ = 0;

	tex_name_ = 0;

	num_vertices_ = 0;

	has_color_ = false;
	has_vertex_ = false;
	has_normal_ = false;
	has_tex_ = false;
	has_img_ = false;
}

void ViewObj::updateVBO(GLenum draw_mode, GLenum Render_mode,const unsigned int &nvertices, float* vertices,
	float* normals, float *colors, float *tex_coords,
	unsigned char *img, unsigned int iwidth, unsigned int iheight )
{
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::cout << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
	}

	releaseVBO();
	initVBO();

	render_mode_ = Render_mode;
	if ( nvertices == 0 || vertices == NULL )
	{
		return;
	}

	num_vertices_ = nvertices;
	has_vertex_ = true;

	glGenBuffers(1, &vbo_buffer_vertices_);
	
	glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer_vertices_);
	
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)* 3*nvertices, vertices, draw_mode);

	if ( normals!=NULL )
	{
		has_normal_ = true;
		glGenBuffers(1, &vbo_buffer_normals_);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer_normals_);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)* 3 * nvertices, normals, draw_mode);
	}
	
	if ( colors!=NULL )
	{
		has_color_ = true;
		glGenBuffers(1, &vbo_buffer_colors_);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer_colors_);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)* 3 * nvertices, colors, draw_mode);
	}

	if ( tex_coords != NULL )
	{
		has_tex_ = true;
		glGenBuffers(1, &vbo_buffer_tex_);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer_tex_);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)* 2 * nvertices, tex_coords, draw_mode);
	}

	if ( img!=NULL && iwidth !=0 && iheight !=0 )
	{
		has_img_ = true;
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glGenTextures(1, &tex_name_);
		glBindTexture(GL_TEXTURE_2D, tex_name_);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR/*GL_NEAREST*/);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR/*GL_NEAREST*/);
		//std::cout << texture_->tex_sizes_[2 * i] << "\t" << texture_->tex_sizes_[1 + 2 * i] << std::endl;
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, iwidth, iheight, 0,
			GL_RGB, GL_UNSIGNED_BYTE, img);
	}
}


void ViewObj::renderVBO()
{
	if ( !has_vertex_  )	return;
	if ( has_color_ )	glEnable(GL_COLOR_MATERIAL);
	if ( has_normal_ )	glEnable(GL_LIGHTING);
	if ( has_tex_ && has_img_ )
	{
		glEnable(GL_TEXTURE_2D);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		glBindTexture(GL_TEXTURE_2D, tex_name_);
	}

	if ( render_mode_ == GL_LINE || render_mode_ == GL_LINES || render_mode_ == GL_LINE_STRIP )
	{
		glEnable(GL_LINE_SMOOTH);
		glLineWidth(3);
		glEnable(GL_BLEND);
		glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}


	glEnableClientState(GL_VERTEX_ARRAY);
	if ( has_normal_ )	glEnableClientState(GL_NORMAL_ARRAY);
	if (has_color_) glEnableClientState(GL_COLOR_ARRAY);
	if (has_tex_ && has_img_) glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer_vertices_);
	glVertexPointer(3, GL_FLOAT, sizeof(float)* 3, (void *)0);

	if ( has_normal_ )
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer_normals_);
		glNormalPointer(GL_FLOAT, sizeof(float)* 3, (void *)0);
	}

	if ( has_color_ )
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer_colors_);
		glColorPointer(3, GL_FLOAT, sizeof(float)* 3, (void *)0);
	}
	
	if ( has_img_ && has_tex_ )
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer_tex_);
		glTexCoordPointer(2, GL_FLOAT, sizeof(float)* 2, (void *)0);
	}

	

	glDrawArrays(render_mode_, 0, num_vertices_);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisableClientState(GL_VERTEX_ARRAY);


	if (render_mode_ == GL_LINE || render_mode_ == GL_LINES || render_mode_ == GL_LINE_STRIP)
	{
		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_BLEND);
	}

	if (has_normal_)glDisableClientState(GL_NORMAL_ARRAY);
	if (has_color_) glDisableClientState(GL_COLOR_ARRAY);
	if (has_tex_ && has_img_) glDisableClientState(GL_TEXTURE_COORD_ARRAY);


	if ( has_img_ && has_tex_ )	glDisable(GL_TEXTURE_2D);
	if (has_normal_) glDisable(GL_LIGHTING);
	if ( has_color_ ) glDisable(GL_COLOR_MATERIAL);
}

void ViewObj::releaseVBO()
{
	releaseVBObuffer(vbo_buffer_colors_);
	releaseVBObuffer(vbo_buffer_normals_);
	releaseVBObuffer(vbo_buffer_tex_);
	releaseVBObuffer(vbo_buffer_vertices_);

	if ( tex_name_!=0 )
	{
		glDeleteTextures(1, &tex_name_);
		tex_name_ = 0;
	}
}

#ifndef MSNHNETOPENGL_H
#define MSNHNETOPENGL_H

#ifdef USE_OPENGL

#ifdef __ANDROID__
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl31.h>
#else

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/glew.h>
#endif

#endif

#include <Msnhnet/utils/MsnhException.h>

#ifndef CHECK_GL_ERROR
#define CHECK_GL_ERROR \
    {                               \
        GLenum error = glGetError();\
        if (GL_NO_ERROR != error)   \
        {                           \
            throw Msnhnet::Exception(1,"Opengl error occured",__FILE__,__LINE__,__FUNCTION__);\
        }                           \
    }
#endif

#endif
#endif 


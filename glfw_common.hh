
#include <iostream>

#define GL_GLEXT_PROTOTYPES 1

#include <GL/gl.h>
#include <GL/glut.h>
//#include "GL/glext.h"

#include "GLFW/glfw3.h"
#include "linmath.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}  

struct vertice2D{
    float x, y;
    float r, g, b;
};

struct vertice3D{
    float x, y, z;
    float r, g, b;
    void print(){
        std::cout<<x<<","<<y<<","<<z<<std::endl;
        std::cout<<r<<","<<g<<","<<b<<std::endl;
    }
};
 

static const char* vertex_shader_text =
"#version 110\n"
"uniform mat4 MVP;\n"
"attribute vec3 vCol;\n"
"attribute vec3 vPos;\n"
"varying vec3 color;\n"
"void main()\n"
"{\n"
"    gl_Position = MVP * vec4(vPos, 1.0);\n"
"    //color = vCol;\n"
"    color = vec3(1.0, 0.0, 0.0);\n"
"}\n";
 
static const char* fragment_shader_text =
"#version 110\n"
"varying vec3 color;\n"
"void main()\n"
"{\n"
"    gl_FragColor = vec4(color, 1.0);\n"
"}\n";


static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}
 
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        //glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}
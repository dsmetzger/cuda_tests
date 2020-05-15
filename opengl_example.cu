// (Minimal) OpenGL to CUDA PBO example
// Maps the default OpenGL depth buffer to CUDA using GL_PIXEL_PACK_BUFFER_ARB
// Purpose of this example is to evaluate why depth transfer is so slow
// Play around with the example by commenting/uncommenting code in lines 77 ff. and in lines 110/112
//
// In order to reproduce the issue, you require:
//  - CUDA (tested with CUDA toolkit 7.5)
//  - GLEW (a version with support for GL_KHR_debug)
//  - (e.g.) freeglut (we need an OpenGL Debug context!)
//
// On Ubuntu 14.04, this example then compiles with the following command line
//  - nvcc main.cu -lglut -lGLEW -lGL
//

#include <assert.h>
#include <stdio.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>

#include <cuda_gl_interop.h>

#define WIDTH  800
#define HEIGHT 800

#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line){
  if(cudaSuccess != err) {
    printf("%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}


// Some dummy kernel to prevent optimizations
__global__ void kernel(unsigned*)
{
}

// Debug callback for use with GL_KHR_debug
void debug_callback_func(
        GLenum          /*source*/,
        GLenum          type,
        GLuint          /*id*/,
        GLenum          severity,
        GLsizei         /*length*/,
        const GLchar*   message,
        GLvoid*         /*user_param*/
        )
{
    printf("%s\n", message);
}


// gl2cuda maps the opengl default frame buffer to cuda
void gl2cuda(int pitch, int h, GLenum format, GLenum type)
{
    GLuint pbo = 0;
    cudaGraphicsResource_t resource = 0;
    void* device_ptr = 0;


    // Setup the PBO and register with CUDA
    glGenBuffers(1, &pbo);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, pitch * h, 0, GL_STREAM_COPY);
    cutilSafeCall( cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsRegisterFlagsReadOnly) );


    // Let's say format is GL_DEPTH_STENCIL. The following is a workaround
    // which provides me with the correct results and is fast but is of course
    // only viable when I don't need the default color buffer anymore afterwards
//  assert(format == GL_DEPTH_STENCIL);
//  glCopyPixels(0, 0, WIDTH, HEIGHT, GL_DEPTH_STENCIL_TO_RGBA_NV);
//  format = GL_BGRA;
//  type = GL_UNSIGNED_BYTE;

    glReadPixels(0, 0, WIDTH, HEIGHT, format, type, 0);


    // Map the graphics resource
    cutilSafeCall( cudaGraphicsMapResources(1, &resource) );
    size_t size = 0;
    cutilSafeCall( cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, resource) );
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);


    // "Use" the data
    kernel<<<1, 1>>>((unsigned*)device_ptr);


    // Unmap and unregister the graphics resource
    cutilSafeCall( cudaGraphicsUnmapResources(1, &resource) );
    cutilSafeCall( cudaGraphicsUnregisterResource(resource) );

    // Delete the PBO
    glDeleteBuffers(1, &pbo);
}

// Display function, issues gl2cuda
void display_func()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Readback with depth/stencil is achingly slow (alas you employ the workaround from line 77 ff.)
    gl2cuda(WIDTH * sizeof(unsigned), HEIGHT, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8);
    // Readback of colors (for comparison) is as fast as expected
//  gl2cuda(WIDTH * sizeof(unsigned), HEIGHT, GL_BGRA, GL_UNSIGNED_BYTE);

    glutSwapBuffers();
}

int main(int argc, char** argv)
{
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInit(&argc, argv);
    // Need freeglut for GLUT_DEBUG!
    glutInitContextFlags(GLUT_DEBUG);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL | GLUT_DOUBLE);
    glutCreateWindow("Depth readback example");

    glewInit();

    // Init GL debug callback to show performance issues
    if (GLEW_KHR_debug)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        glDebugMessageCallback((GLDEBUGPROC)debug_callback_func, 0);
    }
    else
    {
        printf("No GLEW_KHR_debug!");
    }

    glutDisplayFunc(display_func);
    glutMainLoop();
}

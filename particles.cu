/*
*/

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glfw_common.hh"

#include <cuda.h>
#include <cuda_runtime.h>
//#include "cutil.h"
#include <cuda_gl_interop.h>



//#define GL_GLEXT_PROTOTYPES

//#define GLFW_INCLUDE_GLEXT

__device__ void mult(float3 & out, const float3 & vec, const float mult){
    out.x = vec.x*mult;
    out.y = vec.y*mult;
    out.z = vec.z*mult;
}

__device__ void mult_add(float3 & out, const float3 & mult1, const float3 mult2){
    out.x += mult1.x*mult2.x;
    out.y += mult1.y*mult2.y;
    out.z += mult1.z*mult2.z;
}

__device__ float3 sub(const float3 & a, const float3 & b){
    return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);
}

__device__ void zero(float3 & out){
    out.x = 0;
    out.y = 0;
    out.z = 0;
}

__device__ void bound(float3 & cPos, const float3 bound = make_float3(1.0f,1.0f,1.0f)){
    if (cPos.x>bound.x){
        cPos.x = -bound.x + fmod(cPos.x, bound.x);
    }else if (cPos.x<-bound.x){
        cPos.x = bound.x + fmod(cPos.x, -bound.x);
    }

    if (cPos.y>bound.y){
        cPos.y = -bound.y + fmod(cPos.y, bound.y);
    }else if (cPos.y<-bound.y){
        cPos.y = bound.y + fmod(cPos.y, -bound.y);
    }

    if (cPos.z>bound.z){
        cPos.z = -bound.z + fmod(cPos.z, bound.z);
    }else if (cPos.z<-bound.z){
        cPos.z = bound.z + fmod(cPos.z, -bound.z);
    }
}

__device__ float getDistance(float3 & Dist){
    return sqrt(pow(Dist.x,2)+pow(Dist.y,2)+pow(Dist.z,2));
}

__device__ void inverse_squared_law(float3 & Force, float3 Dist, const float mult=0.01){
    float powX = pow(Dist.x,2);
    float powY = pow(Dist.y,2);
    float powZ = pow(Dist.z,2);

    float distance = sqrt(powX+powY+powZ);
    float force = mult/pow(distance,2);
    Force.x += force * Dist.x / distance;
    Force.y += force * Dist.y / distance;
    Force.z += force * Dist.z / distance;
}

__global__ void loop(float3 * cDebug, float3 * cPos, float3 * cVel, float3 * cAccel, float3 * cForce, const float mass = .1, const float dt = .01, const int particles = 1024){
    int idx = threadIdx.x;

    //init
    const float3 dtv = make_float3(dt,dt,dt);
    const float3 massv = make_float3(mass,mass,mass);

    //calc force
    zero(cForce[idx]);
    for (int x=0;x<particles;x++){
        if (x!=idx){
            inverse_squared_law(cForce[idx], sub(cPos[idx], cPos[x]));
        }
    }
    

    //calc position
    mult(cAccel[idx], cForce[idx], 1.0/mass);
    mult_add(cVel[idx], cAccel[idx], dtv);
    mult_add(cPos[idx], cVel[idx], dtv);

    bound(cPos[idx]);
}

__global__ void send_to_opengl(float3 * cPos, float3 * cGraph){
    int idx = threadIdx.x;
    int outIdx = 1*threadIdx.x;
    //if (idx == 0)
    {
        cGraph[outIdx].x = cPos[idx].x;
        cGraph[outIdx].y = cPos[idx].y;
        cGraph[outIdx].z = cPos[idx].z;
        cGraph[outIdx+1].x = 1.0;
        cGraph[outIdx+1].y = 1.0;
        cGraph[outIdx+1].z = 1.0;

        cGraph[outIdx+2].x = cPos[idx].x+.1; 
        cGraph[outIdx+2].y = cPos[idx].y+.1;
        cGraph[outIdx+2].z = cPos[idx].z;
        cGraph[outIdx+3].x = 1.0;
        cGraph[outIdx+3].y = 1.0;
        cGraph[outIdx+3].z = 1.0;

        cGraph[outIdx+4].x = cPos[idx].x+.1;
        cGraph[outIdx+4].y = cPos[idx].y-.1;
        cGraph[outIdx+4].z = cPos[idx].z;
        cGraph[outIdx+5].x = 1.0;
        cGraph[outIdx+5].y = 1.0;
        cGraph[outIdx+5].z = 1.0;
    }
}



using namespace std;

int main(int argc, char ** argv) {
    //Constans
    srand(time(NULL));
    const int ARRAY_SIZE = 1024;
    //const int FLOAT_SIZE = sizeof(float);
    //const int ARRAY_BYTES = ARRAY_SIZE * FLOAT_SIZE;
    const int DIMENSIONS = 3;

    cudaGraphicsResource_t resource = 0;

    int DEBUG = 0;

    const float mass=.1;
    const float dt = .01;

    //vertice2D vertices[3] ={{ -0.6f, -0.4f, 1.f, 0.f, 0.f },{  0.6f, -0.4f, 0.f, 1.f, 0.f },{   0.f,  0.6f, 0.f, 0.f, 1.f }};
    vertice3D vertices[ARRAY_SIZE];

    //opengl pointers
    GLuint vertex_buffer, vertex_shader, fragment_shader, program;
    GLint mvp_location, vpos_location, vcol_location;

    glfwSetErrorCallback(error_callback);
 
    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glViewport(0, 0, 800, 600);

    glfwSwapInterval(1);
    
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  

    // declare GPU memory pointers
    float3 * cDebug;
    float3 * cPos;
    float3 * cVel;
    float3 * cAccel;
    float3 * cForce;

    cudaMallocManaged(&cDebug, DIMENSIONS*ARRAY_SIZE*sizeof(float));//memory usable by host and gpu
    cudaMallocManaged(&cPos, DIMENSIONS*ARRAY_SIZE*sizeof(float));//memory usable by host and gpu
    cudaMallocManaged(&cVel, DIMENSIONS*ARRAY_SIZE*sizeof(float));//memory usable by host and gpu
    cudaMallocManaged(&cAccel, DIMENSIONS*ARRAY_SIZE*sizeof(float));//memory usable by host and gpu
    cudaMallocManaged(&cForce, DIMENSIONS*ARRAY_SIZE*sizeof(float));//memory usable by host and gpu

    // initialize
    for (int x = 0; x<ARRAY_SIZE; ++x){
        cPos[x].x=float(rand())/float(RAND_MAX)-.5;
        cPos[x].y=float(rand())/float(RAND_MAX)-.5;
        cPos[x].z=float(rand())/float(RAND_MAX)-.5;

        //cForce[x].x=0;
    }

/*
    GLuint vertexArray;
    glGenBuffers( 1,&vertexArray);
    glBindBuffer( GL_ARRAY_BUFFER, vertexArray);
    glBufferData( GL_ARRAY_BUFFER, DIMENSIONS*ARRAY_SIZE*sizeof(float), NULL, GL_DYNAMIC_COPY );
    cudaGLRegisterBufferObject( vertexArray );


    GLuint gl_buffer, gl_target;
    
    struct cudaGraphicsResource *vbo_res;
    cudaGraphicsGLRegisterImage(&vbo_res, gl_buffer, gl_target, cudaGraphicsRegisterFlagsSurfaceLoadStore);
*/

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    //glGenBuffers(1, &pbo);
    //glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
    //glBufferData(GL_PIXEL_PACK_BUFFER, pitch * h, 0, GL_STREAM_COPY);
    cudaGraphicsGLRegisterBuffer(&resource, vertex_buffer, cudaGraphicsRegisterFlagsWriteDiscard);

    cudaGraphicsMapResources(1, &resource);
    void* device_ptr = 0;
    size_t size = 0;//shared size bytes  
    cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, resource);
    std::cout<<"shared size bytes "<<size<<std::endl;

    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
    glCompileShader(vertex_shader);
 
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
    glCompileShader(fragment_shader);

    program = glCreateProgram();
    glAttachShader(program, vertex_shader);//vertex runs once per triangle.
    glAttachShader(program, fragment_shader);//fragment runs once per pixel.
    glLinkProgram(program);

    mvp_location = glGetUniformLocation(program, "MVP");
    vpos_location = glGetAttribLocation(program, "vPos");
    vcol_location = glGetAttribLocation(program, "vCol");

    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (void*) 0);
    glEnableVertexAttribArray(vcol_location);
    glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (void*) (sizeof(float) * 2));

    //init<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, mass, dt);

    /*
    for (int x = 0; x<5; ++x){
        if (DEBUG==1){
            cudaDeviceSynchronize();
            cout<< cPos[0].x<<","<<cPos[0].y<<","<<cPos[0].z<<endl;
        }

        loop<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, cAccel, cForce, mass, dt, ARRAY_SIZE);
        //graph<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, mass, dt);
    }
    */

    int iters = 0;
    while(!glfwWindowShouldClose(window))
    {

        loop<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, cAccel, cForce, mass, dt, ARRAY_SIZE);
        send_to_opengl<<<1, ARRAY_SIZE>>>(cPos, (float3*)device_ptr);
        cudaDeviceSynchronize();//TODO:replace with right synch

        //vertice3D* tmpPtr = (vertice3D*)device_ptr;
        //tmpPtr[0].print();

        float ratio;
        int width, height;
        mat4x4 m, p, mvp;
 
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
 
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
 
        mat4x4_identity(m);
        mat4x4_rotate_Z(m, m, (float) glfwGetTime());
        mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        mat4x4_mul(mvp, p, m);
        
        glUseProgram(program);
        glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*) mvp);
        glDrawArrays(GL_TRIANGLES, 0, 3);
 
        glfwSwapBuffers(window);
        glfwPollEvents();

        ++iters;
    }
    std::cout << iters << std::endl;

    glfwDestroyWindow(window);
    glfwTerminate();

    cudaDeviceSynchronize();

    for (int x = 0; x<min(64,ARRAY_SIZE); ++x){
        //cout<< cPos[x].x<<","<<cPos[x].y<<","<<cPos[x].z<<endl;
    }

    cudaFree(cDebug);

    return 0;
}

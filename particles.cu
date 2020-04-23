/*
*/
#include <stdio.h>
#include <iostream>

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

__device__ void bound(float3 & cPos, const float3 bound = make_float3(1.0f,1.0f,1.0f)){
    //cPos
}

__global__ void loop(float3 * cDebug, float3 * cPos, float3 * cVel, float3 * cAccel, float3 * cForce, const float mass = .1, const float dt = .01){
    int idx = threadIdx.x;

    //init
    const float3 dtv = make_float3(dt,dt,dt);
    const float3 massv = make_float3(mass,mass,mass);

    //calc force


    //calc position
    mult(cAccel[idx], cForce[idx], 1.0/mass);
    mult_add(cVel[idx], cAccel[idx], dtv);
    mult_add(cPos[idx], cVel[idx], dtv);

    bound(cPos[idx]);
}

using namespace std;

int main(int argc, char ** argv) {
    const int ARRAY_SIZE = 64;
    //const int FLOAT_SIZE = sizeof(float);
    //const int ARRAY_BYTES = ARRAY_SIZE * FLOAT_SIZE;
    const int DIMENSIONS = 3;

    const float mass=.1;
    const float dt = .01;

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
        cForce[x].x=100;
    }
    
    
    //init<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, mass, dt);

    for (int x = 0; x<5; ++x){
       loop<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, cAccel, cForce, mass, dt);
       //graph<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, mass, dt);
    }

    cudaDeviceSynchronize();

    for (int x = 0; x<min(64,ARRAY_SIZE); ++x){
        cout<< cPos[x].x<<","<<cPos[x].y<<","<<cPos[x].z<<endl;
    }

    cudaFree(cDebug);

    return 0;
}

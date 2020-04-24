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

using namespace std;

int main(int argc, char ** argv) {
    srand(time(NULL));
    const int ARRAY_SIZE = 1024;
    //const int FLOAT_SIZE = sizeof(float);
    //const int ARRAY_BYTES = ARRAY_SIZE * FLOAT_SIZE;
    const int DIMENSIONS = 3;

    int DEBUG = 1;

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
        cPos[x].x=float(rand())/float(RAND_MAX)-.5;
        cPos[x].y=float(rand())/float(RAND_MAX)-.5;
        cPos[x].z=float(rand())/float(RAND_MAX)-.5;

        //cForce[x].x=0;
    }
    
    
    //init<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, mass, dt);

    for (int x = 0; x<5; ++x){
        if (DEBUG==1){
            cudaDeviceSynchronize();
            cout<< cPos[0].x<<","<<cPos[0].y<<","<<cPos[0].z<<endl;
        }

        loop<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, cAccel, cForce, mass, dt, ARRAY_SIZE);
        //graph<<<1, ARRAY_SIZE>>>(cDebug, cPos, cVel, mass, dt);


    }

    cudaDeviceSynchronize();

    for (int x = 0; x<min(64,ARRAY_SIZE); ++x){
        //cout<< cPos[x].x<<","<<cPos[x].y<<","<<cPos[x].z<<endl;
    }

    cudaFree(cDebug);

    return 0;
}

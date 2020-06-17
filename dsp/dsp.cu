

__global__ void DFT(float *x_re,float *x_im,float *X_re,float *X_im,int size)
{
    size_t id = get_global_id(0);   // each thread works on an element
    X_re[id] = 0.0;
    X_im[id] = 0.0;
    for (int i = 0; i < size; i++) {
        X_re += (x_re[id] * cospif(2 * id / size) - x_im[id] + sinpif(2 * id / size));
        X_im += (x_re[id] * sinpif(2 * id / size) + x_im[id] + cospif(2 * id / size));
    }
}

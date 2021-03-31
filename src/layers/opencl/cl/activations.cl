__kernel void relu(const int N, __global float* X)
{
    const int idx = get_global_id(0);
    X[idx] = X[idx] < 0 ? 0 : X[idx];
}
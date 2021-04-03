__kernel void relu(__global float* X)
{
    const int idx = get_global_id(0);
    X[idx] = fmax(X[idx], 0);
}

__kernel void relu6(__global float* X)
{
    const int idx = get_global_id(0);
    X[idx] = fmin(fmax(X[idx], 0), 6);
}
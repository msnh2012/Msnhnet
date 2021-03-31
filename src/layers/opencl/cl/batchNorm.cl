__kernel void BatchNorm(const int inSize,
                        const __global float* src,
                        const __global float* biases,
                        const __global float* Scales,
                        const __global float* rollMean,
                        const __global float* rollVariance,
                        __global float* dst)
{
    const int chIdx = get_global_id(0);
    float sqrtVar = native_sqrt(rollVariance[chIdx] + 0.00001f);
    float a = biases[chIdx] - Scales[chIdx] * rollMean[chIdx] / sqrtVar;
    float b = Scales[chIdx] / sqrtVar;

    // printf("chIndex = %d,  a = %f   b = %f\n", chIdx, a, b);
    // printf("chIndex = %d,  biases = %f   Scales = %f    rollMean = %f    rollVariance = %f \n", chIdx, biases[chIdx], Scales[chIdx], rollMean[chIdx], rollVariance[chIdx]);


    for (int i = 0; i < inSize; i++) {
        dst[chIdx * inSize + i] = b * src[chIdx * inSize + i] + a;
    }
}

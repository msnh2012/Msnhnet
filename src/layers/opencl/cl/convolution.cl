__kernel void IM2COL_GEMM(const int M, const int N, const int K, const __global float* A, 
                      const __global float* B, __global float* C)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    // if (globalCol == 0 && globalRow == 0) {
    //     printf("A : %f   \t%f   \t%f   \t%f   \t%f\n", A[9], A[10], A[11], A[16], A[17]);
    //     printf("B : %f   \t%f   \t%f   \t%f   \t%f\n", B[0], B[1 * N], B[2 * N], B[3 * N], B[4 * N]);
    // }

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[globalRow * K + k] * B[k * N + globalCol];

        // if (globalCol == 0 && globalRow == 0 && k < 100) {
        //     printf("acc : %f\n", acc);
        // }
    }
    C[globalRow * N + globalCol] = acc;
    
    // if (globalCol == 0 && globalRow == 0) {
    //     printf("acc = %f\n", acc);
    // }
}

__kernel void IM2COL_GEMM_c(const int M, const int N, const int K, const __global float* A, 
                      const __global float* B, __global float* C)
{
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    // if (globalCol == 0 && globalRow == 0) {
    //     printf("A : %f   \t%f   \t%f   \t%f   \t%f\n", A[9], A[10], A[11], A[16], A[17]);
    //     printf("B : %f   \t%f   \t%f   \t%f   \t%f\n", B[0], B[1 * N], B[2 * N], B[3 * N], B[4 * N]);
    // }

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[globalRow * K + k] * B[globalCol * K + k];

        // if (globalCol == 0 && globalRow == 0 && k < 100) {
        //     printf("acc : %f\n", acc);
        // }
    }
    C[globalRow * N + globalCol] = acc;
    
    if (globalCol == 0 && globalRow == 0) {
        printf("acc = %f\n", acc);
    }
}


bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return (unsigned int)(a) < (unsigned int)(b);
}

__kernel void im2col(const int inWidth,
    const int inHeight,
    const int outWidth,
    const int outHeight,
    const int filterW,
    const int filterH,
    const int paddingX,
    const int paddingY,
    const int strideX,
    const int strideY,
    const int dilationX,
    const int dilationY,
    const __global float* src,
    __global float* dst)
{
    const int channelIdx = get_global_id(0);

    const __global float* src_0 = src + channelIdx * inWidth * inHeight;
    __global float* dst_0 = dst + channelIdx * filterW * filterH * outWidth * outHeight;

    for (int filterRow = 0; filterRow < filterH; filterRow++) {
        for (int filterCol = 0; filterCol < filterW; filterCol++) {
            int inputRow = -paddingY + filterRow * dilationY;
            for (int outputRow = 0; outputRow < outHeight; ++outputRow)
            {
                int row = -paddingY + outputRow * strideY + filterRow * dilationY;
                if (!is_a_ge_zero_and_a_lt_b(row, inHeight))
                //if (true)
                {
                    for (int outputCol = 0; outputCol < outWidth; ++outputCol)
                    {
                        *dst_0 = 0;
                        dst_0++;
                    }
                }
                else
                {
                    int inputCol = -paddingX + filterCol * dilationX;
                    for (int outputCol = 0; outputCol < outWidth; ++outputCol)
                    {
                        int col = -paddingX + outputCol * strideX + filterCol * dilationX;
                        if (is_a_ge_zero_and_a_lt_b(col, inWidth)) {
                        //if (true) {

                            int ori_idx = row * inWidth + col;

                            *dst_0 = src_0[ori_idx];
                        }
                        else {
                            *dst_0 = 0;
                        }
                        dst_0++;
                    }
                }
            }
        }
    }
}


__constant static const float ktm[8][3] = {
    {1.0f,      0.0f,      0.0f},
    {-2.0f / 9, -2.0f / 9, -2.0f / 9},
    {-2.0f / 9, 2.0f / 9, -2.0f / 9},
    {1.0f / 90, 1.0f / 45, 2.0f / 45},
    {1.0f / 90, -1.0f / 45, 2.0f / 45},
    {1.0f / 45, 1.0f / 90, 1.0f / 180},
    {1.0f / 45, -1.0f / 90, 1.0f / 180},
    {0.0f, 0.0f, 1.0f}
};

__kernel void FilterTransform(const __global float* filter, __global float* filterWino)
{
    const int index = get_global_id(0);

    int filterOffset = index * 9;
    int filterWinoOffset = index * 64;

    const __global float* k0 = filter + filterOffset;
    const __global float* k1 = filter + filterOffset + 3;
    const __global float* k2 = filter + filterOffset + 6;

    __global float* filterWino_ = filterWino + filterWinoOffset;

    float tmpG[8][3];    // tmp = G*g
    for (int i = 0; i < 8; i++) {
        tmpG[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
        tmpG[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
        tmpG[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
    }

    for (int i = 0; i < 8; i++) {
        float* tmpPtr = &tmpG[i][0];
        for (int j = 0; j < 8; j++) {
            filterWino_[i * 8 + j] = tmpPtr[0] * ktm[j][0] + tmpPtr[1] * ktm[j][1] + tmpPtr[2] * ktm[j][2];
        }
    }
}



__kernel void convWinograd(const int inWidth,
                              const int inHeight,
                              const int inChannel,
                              const int outWidth,
                              const int outHeight,
                              const __global float* src,
                              const __global float* filter,
                              __global float* dst)
{
     const int outChIdx = get_global_id(0);
    __global float* dst0 = dst + outChIdx * outWidth * outHeight;

    float tmpV[8][8];
    float tmpA[6][8];
    for (int k = 0; k < inChannel; k++){
        for(int i = 0; i < outHeight / 6; i++){
            for(int j = 0; j < outWidth / 6; j++){
                const __global float* r0 = src + k * inWidth * inHeight + i * 6 * inWidth + j * 6;
                __global float* outputPtr = dst0 + i * 6 * outWidth + j * 6;

                for (int m = 0; m < 8; m++) {
                    tmpV[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                    tmpV[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                    float t1 = (r0[2] + r0[6] - r0[4] * 4.25f);
                    float t2 = (r0[1] + r0[5] - r0[3] * 4.25f);

                    tmpV[1][m] = t1 + t2;
                    tmpV[2][m] = t1 - t2;

                    float t3 = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                    float t4 = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);
                    tmpV[3][m] = t3 + t4;
                    tmpV[4][m] = t3 - t4;

                    float t5 = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                    float t6 = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                    tmpV[5][m] = t5 + t6;
                    tmpV[6][m] = t5 - t6;

                    r0 += inWidth;
                }
                
                for (int m = 0; m < 8; m++) {
                    const __global float* filter0 = filter + outChIdx * inChannel * 8 * 8 + k * 8 * 8 + m * 8;
                    float d0 = (tmpV[m][0] - tmpV[m][6] + (tmpV[m][4] - tmpV[m][2]) * 5.25f) * filter0[0];
                    float d7 = (tmpV[m][7] - tmpV[m][1] + (tmpV[m][3] - tmpV[m][5]) * 5.25f) * filter0[7];
                    
                    float t1 =  (tmpV[m][2] + tmpV[m][6] - tmpV[m][4] * 4.25f);
                    float t2 =  (tmpV[m][1] - tmpV[m][3] * 4.25f + tmpV[m][5]);
                    float d1 = (t1 + t2) * filter0[1];
                    float d2 = (t1 - t2) * filter0[2];

                    float t3 = (tmpV[m][6] + tmpV[m][2] * 0.25f - tmpV[m][4] * 1.25);
                    float t4 = (tmpV[m][1] * 0.5f - tmpV[m][3] * 2.5f + tmpV[m][5] * 2.f);
                    float d3 = (t3 + t4) * filter0[3];
                    float d4 = (t3 - t4) * filter0[4];

                    float t5 = (tmpV[m][6] + (tmpV[m][2] - tmpV[m][4] * 1.25f) * 4.f);
                    float t6 = (tmpV[m][1] * 2.f - tmpV[m][3] * 2.5f + tmpV[m][5] * 0.5f);

                    float d5 = (t5 + t6) * filter0[5];
                    float d6 = (t5 - t6) * filter0[6];

                    t1 = d1 + d2;
                    t2 = d1 - d2;

                    t3 = d3 + d4;
                    t4 = d3 - d4;

                    t5 = d5 + d6;
                    t6 = d5 - d6;


                    tmpA[0][m] = d0 + t1 + t3 + t5 * 32;
                    tmpA[2][m] = t1 + t3 * 4 + t5 * 8;
                    tmpA[4][m] = t1 + t3 * 16 + t5 + t5;

                    tmpA[1][m] = t2 + t4 + t4 + t6 * 16;
                    tmpA[3][m] = t2 + t4 * 8 + t6 * 4;
                    tmpA[5][m] = d7 + t2 + t4 * 32 + t6;

                    
                }

                for(int m = 0; m < 6; m++){

                    const float* tmp0 = tmpA[m];

                    float t1 = tmp0[1] + tmp0[2];
                    float t2 = tmp0[1] - tmp0[2];

                    float t3 = tmp0[3] + tmp0[4];
                    float t4 = tmp0[3] - tmp0[4];

                    float t5 = tmp0[5] + tmp0[6];
                    float t6 = tmp0[5] - tmp0[6];

                    outputPtr[0] += tmp0[0] + t1 + t3 + t5 * 32;
                    outputPtr[2] += t1 + t3 * 4 + t5 * 8;
                    outputPtr[4] += t1 + t3 * 16 + t5 + t5;

                    outputPtr[1] += t2 + t4 + t4 + t6 * 16;
                    outputPtr[3] += t2 + t4 * 8 + t6 * 4;
                    outputPtr[5] += tmp0[7] + t2 + t4 * 32 + t6;
                    outputPtr += outWidth;
                }
            }
        }
    }

}
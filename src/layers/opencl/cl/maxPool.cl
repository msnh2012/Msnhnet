__kernel void maxPoolGeneral(const int inWidth, 
                             const int inHeight, 
                             const int inChannel, 
                             const int outWidth, 
                             const int outHeight, 
                             const int outChannel,
                             const int filterW, 
                             const int filterH, 
                             const int strideX, 
                             const int strideY, 
                             const int paddingX, 
                             const int paddingY,
                             const __global float* src, 
                             __global float* dst)
{
    const int globalCol = get_global_id(0);
    const int globalRow = get_global_id(1);
    const int globalChn = get_global_id(2);


    int idxout = outWidth * outHeight * globalChn + outWidth * globalRow + globalCol;
    float maxF = -FLT_MAX;
    int inputOffset = inWidth * inHeight * globalChn;

    for (int i = 0; i < filterH; i++) {
        for (int j = 0; j < filterW; j++) {
            
            int inCol = globalCol * strideX + j - paddingX;
            int inRow = globalRow * strideY + i - paddingY;
            if (inCol >= inWidth + paddingX || inRow >= inHeight + paddingY || inCol < -paddingX || inRow < -paddingY)
                continue;
            bool valid = (inCol < inWidth && inRow < inHeight && inCol >= 0 && inRow >= 0);
            int inIdx = inRow * inWidth + inCol + inputOffset;

            float val = valid ? src[inIdx] : -FLT_MAX;
            maxF = val > maxF ? val : maxF;
        }
    }
    dst[idxout] = maxF;
}

__kernel void localAverage(
    const int inWidth,
    const int inHeight,
    const int outWidth,
    const int outHeight,
    const int filterWidth,
    const int filterHeight,
    const int strideX,
    const int strideY,
    const int paddingX,
    const int paddingY,
    const __global float* src,
    __global float* dst)
{
    
    int chIdx = get_global_id(0);

    for (size_t i = 0; i < outHeight; i++)
    {
        for (size_t j = 0; j < outWidth; j++)
        {
            float sum = 0;
            float count = 0;
            for (size_t p = 0; p < filterHeight; p++)
            {
                for (size_t q = 0; q < filterWidth; q++)
                {
                    int inRow = i * strideY + p - paddingY;
                    int inCol = j * strideX + q - paddingX;

                    if (inRow < inHeight && inCol < inWidth && inRow >= 0 && inCol >= 0){
                        sum += src[inWidth * inHeight * chIdx + inRow * inWidth + inCol];
                        count += 1;
                    }
                }
            }
            dst[outWidth * outHeight * chIdx + i * outWidth + j] = sum / count;
        }
    }
}

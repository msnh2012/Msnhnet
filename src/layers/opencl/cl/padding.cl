// __kernel void padding(
//     const int inWidth,
//     const int inHeight,
//     const int top,
//     const int down,
//     const int left,
//     const int right,
//     const float val,
//     const __global float* src,
//     __global float* dst)
// {
//     const int globalInCh = get_global_id(0);
//     int srcOffset = inWidth * inHeight * globalInCh;
//     int dstOffset = (inWidth + left + right) * (inHeight + top * down) * globalInCh;
//     for (int i = 0; i < top; i ++) {
//         for(int j = 0; j < (inWidth + left + right); j++, dstOffset++) {
//             *(dst + dstOffset) = val;
//         }
//     }
//     for (int i = 0; i < inHeight; i++) {
//         for(int j = 0; j < (inWidth + left + right); j++, dstOffset++) {
//             if (j < left || j >= inWidth + left)
//                 *(dst + dstOffset) = val;
//             else {
//                 *(dst + dstOffset) = *(src + srcOffset);
//                 srcOffset++;
//             }
//         }
//     }
//     for (int i = 0; i < down; i ++) {
//         for(int j = 0; j < (inWidth + left + right); j++, dstOffset++) {
//             *(dst + dstOffset) = val;
//         }
//     }
// }

__kernel void padding(
    const int inWidth,
    const int inHeight,
    const int top,
    const int down,
    const int left,
    const int right,
    const float val,
    const __global float* src,
    __global float* dst)
{
    const int newWidth = inWidth + left + right;
    const int newHeight = inHeight + top + down;
    const int globalInCh = get_global_id(0);
    const __global float* src0 = src + inWidth * inHeight * globalInCh;
    __global float* dst0 = dst + newWidth * newHeight * globalInCh;

    for (int i = 0; i < newHeight; i++) {

        int old_row = i - top;

        if (old_row < 0 || old_row >= inHeight) {
            for (int j = 0; j < newWidth; j++) {
                dst0[i * newWidth + j] = val;
            }
            continue;
        }

        for (int j = 0; j < newWidth; j++) {
            int old_col = j - left;
            if (old_col < 0 || old_col >= inWidth) {
                dst0[i * newWidth + j] = val;
            }
            else {
                dst0[i * newWidth + j] = src0[old_row * inWidth + old_col];
            }
        }


    }

}
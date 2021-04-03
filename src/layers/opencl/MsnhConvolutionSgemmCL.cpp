#ifdef USE_OPENCL

#include "Msnhnet/layers/opencl/MsnhConvolutionSgemmCL.h"
#include <fstream> ///////////////////////////////////////////////////


namespace Msnhnet
{
    void ConvolutionSgemmCL::convolutionIm2colSgemmCL(
        cl_mem &src, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_mem &filter, 
        cl_kernel &kernel_cov, 
        cl_kernel &kernel_im2col,
        const int &filterW, 
        const int &filterH, 
        cl_mem &dst, 
        const int &outWidth, 
        const int &outHeight, 
        const int &outChannel, 
        const int& StrideH, 
        const int &StrideW, 
        const int &paddingX, 
        const int &paddingY, 
        const int &dilationW, 
        const int &dilationH)
    {

        std::cout << "convolutionIm2colSgemmCL" << std::endl;

        int m       =  outChannel; 
        int n       =  outWidth * outHeight;
        int k       =  filterW * filterH * inChannel;

        cl_int status = 0;
        // cl_mem imMem = clCreateBuffer(clScheduler::get().context(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, inWidth * inHeight * inChannel * sizeof(float), src, &status);
        // CHECKSTATUS(status, "create imMem");
        cl_mem im_im2col = clCreateBuffer(clScheduler::get().context(),  CL_MEM_READ_WRITE, n * k * sizeof(float), NULL, &status);
        CHECKSTATUS(status, "create im_im2col");
        
        status = clSetKernelArg(kernel_im2col, 0, sizeof(int), (void*) &inWidth);
        status |= clSetKernelArg(kernel_im2col, 1, sizeof(int), (void*) &inHeight);
        status |= clSetKernelArg(kernel_im2col, 2, sizeof(int), (void*) &outWidth);
        status |= clSetKernelArg(kernel_im2col, 3, sizeof(int), (void*) &outHeight);
        status |= clSetKernelArg(kernel_im2col, 4, sizeof(int), (void*) &filterW);
        status |= clSetKernelArg(kernel_im2col, 5, sizeof(int), (void*) &filterH);
        status |= clSetKernelArg(kernel_im2col, 6, sizeof(int), (void*) &paddingX);
        status |= clSetKernelArg(kernel_im2col, 7, sizeof(int), (void*) &paddingY);
        status |= clSetKernelArg(kernel_im2col, 8, sizeof(int), (void*) &StrideW);
        status |= clSetKernelArg(kernel_im2col, 9, sizeof(int), (void*) &StrideH);
        status |= clSetKernelArg(kernel_im2col, 10, sizeof(int), (void*) &dilationW);
        status |= clSetKernelArg(kernel_im2col, 11, sizeof(int), (void*) &dilationH);
        status |= clSetKernelArg(kernel_im2col, 12, sizeof(cl_mem), (void*) &src);
        status |= clSetKernelArg(kernel_im2col, 13, sizeof(cl_mem), (void*) &im_im2col);
        CHECKSTATUS(status, "set kernel_im2col args");
        
        size_t global_[1] = {inChannel};
        size_t local_[1] = {1};
        cl_event eventPoint_;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel_im2col, 1, NULL, global_, local_, 0, NULL, &eventPoint_);
        clWaitForEvents(1, &eventPoint_);
        clReleaseEvent(eventPoint_);
        
        // float* im_im2col_ptr = (float*)clEnqueueMapBuffer(clScheduler::get().queue(), im_im2col, CL_TRUE, CL_MAP_WRITE, 0, n * k * sizeof(float), 0, NULL, NULL, &status);
        // CHECKSTATUS(status, "map im_im2col buffer");
        // float* im_im2col_ptr0 = im_im2col_ptr;
        // // const int Stride = this->_kSizeX *  this->_kSizeY * this->_outHeight * this->_outWidth;

        // for (int cc = 0; cc < inChannel; cc++) 
        // {
        //     const float *src0 = src + cc *  inWidth * inHeight;
        //     for (int filterRow = 0; filterRow < filterH; filterRow++)   
        //     {
        //         for (int filterCol = 0; filterCol < filterW; filterCol++) 
        //         {
        //             int inputRow = -paddingY + filterRow * dilationH;
        //             for (int outputRow = 0; outputRow < outHeight; ++outputRow)
        //             {
        //                 int row = -paddingY + outputRow * StrideH + filterRow * dilationH;
        //                 if (!is_a_ge_zero_and_a_lt_b_cl(row, inHeight)) 
        //                 {
        //                     for (int outputCol = 0; outputCol < outWidth; ++outputCol) 
        //                     {
        //                         *im_im2col_ptr0 = 0;
        //                         im_im2col_ptr0++;
        //                     }
        //                 }
        //                 else
        //                 {
        //                     int inputCol = -paddingX + filterCol * dilationW; 
        //                     for (int outputCol = 0 ; outputCol < outWidth; ++outputCol)
        //                     {
        //                         int col = -paddingX + outputCol * StrideW + filterCol * dilationW;
        //                         if (is_a_ge_zero_and_a_lt_b_cl(col, inWidth)) {
        //                             // int row = -paddingY + outputRow * StrideH + filterRow * dilationH;
        //                             int ori_idx = row * inWidth + col;
        //                             *im_im2col_ptr0 = src0[ori_idx];
        //                         } 
        //                         else {
        //                             *im_im2col_ptr0 = 0;
        //                         }
        //                         im_im2col_ptr0++;
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        // status = clEnqueueUnmapMemObject(clScheduler::get().queue(), im_im2col, im_im2col_ptr, 0, NULL, NULL);
        // CHECKSTATUS(status, "unmap im_im2col buffer");


        status = clSetKernelArg(kernel_cov, 0, sizeof(int), (void*) &m);
        status |= clSetKernelArg(kernel_cov, 1, sizeof(int), (void*) &n);
        status |= clSetKernelArg(kernel_cov, 2, sizeof(int), (void*) &k);
        status |= clSetKernelArg(kernel_cov, 3, sizeof(cl_mem), (void*) &filter);
        status |= clSetKernelArg(kernel_cov, 4, sizeof(cl_mem), (void*) &im_im2col);
        status |= clSetKernelArg(kernel_cov, 5, sizeof(cl_mem), (void*) &dst);
        CHECKSTATUS(status, "set kernel args");

        cl_event eventPoint;
        size_t global[2] = {m, n};
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel_cov, 2, NULL, global, NULL, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);

        CHECKSTATUS(status, "calc matrix mul");
        // status |= clEnqueueReadBuffer(clScheduler::get().queue(), dst, CL_TRUE, 0, m * n * sizeof(float), dest, 0, NULL, NULL);

    }

    void  ConvolutionSgemmCL::conv3x3s1WinogradTransformKenelCL(cl_kernel &kernel, cl_mem &filter, cl_mem &filterWino,const int &inChannel, const int &outChannel){
        cl_int status = 0;
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &filter);
        status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &filterWino);
        CHECKSTATUS(status, "set kernel args");

        size_t global[1] = {inChannel * outChannel};
        size_t local[1] = {1};
        cl_event eventPoint;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel, 1, NULL, global, local, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);
        CHECKSTATUS(status, "transform filter");
    }



    void  ConvolutionSgemmCL::conv3x3s1WinogradCL(
        cl_mem &src, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_mem &filter, 
        cl_kernel &kernel_cov, 
        cl_kernel &kernel_pad,
        cl_mem &dst, 
        const int &outWidth, 
        const int &outHeight, 
        const int &outChannel, 
        const int &paddingX, 
        const int &paddingY)
    {
        int newInWidth = (inWidth + 2 * paddingX - 2 + 5) / 6 * 6 + 2;
        int newInHeight = (inHeight + 2 * paddingY - 2 + 5) / 6 * 6 + 2;
        
        cl_int status = 0;
        cl_mem srcPad = clCreateBuffer(clScheduler::get().context(), CL_MEM_READ_WRITE, newInWidth * newInHeight * inChannel * sizeof(float), NULL, &status);
        
        CHECKSTATUS(status, "create srcPad");

        int left = paddingX;
        int top = paddingY;
        int right = newInWidth - left - inWidth;
        int down = newInHeight - top - inHeight;
        PaddingCL::paddingCL(src, inWidth, inHeight, inChannel, kernel_pad, srcPad, top, down, left, right, 0);

        int dstTempWidth = newInWidth - 2;
        int dstTempHeight = newInHeight - 2;

        cl_mem dstTemp = clCreateBuffer(clScheduler::get().context(), CL_MEM_READ_WRITE, dstTempWidth * dstTempHeight * outChannel * sizeof(float), NULL, &status);

        status = clSetKernelArg(kernel_cov, 0, sizeof(cl_int), (void*)&newInWidth);
        status |= clSetKernelArg(kernel_cov, 1, sizeof(cl_int), (void*)&newInHeight);
        status |= clSetKernelArg(kernel_cov, 2, sizeof(cl_int), (void*)&inChannel);
        status |= clSetKernelArg(kernel_cov, 3, sizeof(cl_int), (void*)&dstTempWidth);
        status |= clSetKernelArg(kernel_cov, 4, sizeof(cl_int), (void*)&dstTempHeight);
        status |= clSetKernelArg(kernel_cov, 5, sizeof(cl_mem), (void*)&srcPad);
        status |= clSetKernelArg(kernel_cov, 6, sizeof(cl_mem), (void*)&filter);
        status |= clSetKernelArg(kernel_cov, 7, sizeof(cl_mem), (void*)&dstTemp);
        if (status != CL_SUCCESS) { std::cout << "set arg failed" << std::endl; }

        size_t global[1] = {outChannel};
        size_t local[1] = {1};
        cl_event eventPoint;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel_cov, 1, NULL, global, local, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);

        PaddingCL::paddingCL(dstTemp, dstTempWidth, dstTempHeight, outChannel, kernel_pad, dst, 0,  outHeight - dstTempHeight, 0, outWidth - dstTempWidth, 0);

        return;
    }

}



#endif

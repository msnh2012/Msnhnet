if(BUILD_USE_GPU MATCHES OFF)
    add_subdirectory(lenet5)

    add_subdirectory(classify)

    add_subdirectory(yolov3)

    add_subdirectory(yolov3tiny)

    add_subdirectory(landmark106)

    if(BUILD_USE_OPENCV MATCHES ON)
        add_subdirectory(yolov3tiny_video)
    elseif(BUILD_USE_MSNHCV_GUI MATCHES ON)
        add_subdirectory(yolov3tiny_video)
    endif()

    add_subdirectory(yolov4)

    add_subdirectory(unet)

    add_subdirectory(fcns)
    
    add_subdirectory(yolov5s)

    add_subdirectory(yolov5m)
    
    add_subdirectory(puma560)

endif()

if(BUILD_USE_GPU MATCHES ON)
    add_subdirectory(classify_gpu)

    add_subdirectory(yolov3_gpu)

    add_subdirectory(yolov3tiny_gpu)

    if(BUILD_USE_OPENCV MATCHES ON)
        add_subdirectory(yolov3tiny_video_gpu)
    elseif(BUILD_USE_MSNHCV_GUI MATCHES ON)
        add_subdirectory(yolov3tiny_video_gpu)
    endif()

    add_subdirectory(yolov4_gpu)

    add_subdirectory(unet_gpu)

    add_subdirectory(deeplabv3_gpu)

    add_subdirectory(yolov5s_gpu)

    add_subdirectory(yolov5m_gpu)
    
    if(BUILD_USE_CUDNN MATCHES ON)

        add_subdirectory(mobilenetv2_yolov3_lite_gpu)
        
        add_subdirectory(mobilenetv2_yolov3_nano_gpu)

        add_subdirectory(yoloface100k_gpu)

        add_subdirectory(yoloface500k_gpu)

        add_subdirectory(unet_gpu_fp16)

        add_subdirectory(yolov3_gpu_fp16)

        add_subdirectory(yolov3tiny_gpu_fp16)

        add_subdirectory(yolov4_gpu_fp16)
    endif()
endif()

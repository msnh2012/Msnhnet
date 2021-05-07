if(BUILD_USE_GPU MATCHES OFF)
    add_subdirectory(deeplearning/lenet5)

    add_subdirectory(deeplearning/classify)

    add_subdirectory(deeplearning/yolov3)

    add_subdirectory(deeplearning/yolov3tiny)

    add_subdirectory(deeplearning/landmark106)

    if(BUILD_USE_OPENCV MATCHES ON)
        add_subdirectory(deeplearning/yolov3tiny_video)
    elseif(BUILD_USE_MSNHCV_GUI MATCHES ON)
        add_subdirectory(deeplearning/yolov3tiny_video)
    endif()

    add_subdirectory(deeplearning/yolov4)

    add_subdirectory(deeplearning/unet)

    add_subdirectory(deeplearning/fcns)
    
    add_subdirectory(deeplearning/yolov5s)

    add_subdirectory(deeplearning/yolov5m)
    
    add_subdirectory(robotics/puma560)

    add_subdirectory(robotics/ur5)

endif()

if(BUILD_USE_GPU MATCHES ON)
    add_subdirectory(deeplearning/classify_gpu)

    add_subdirectory(deeplearning/yolov3_gpu)

    add_subdirectory(deeplearning/yolov3tiny_gpu)

    if(BUILD_USE_OPENCV MATCHES ON)
        add_subdirectory(deeplearning/yolov3tiny_video_gpu)
    elseif(BUILD_USE_MSNHCV_GUI MATCHES ON)
        add_subdirectory(deeplearning/yolov3tiny_video_gpu)
    endif()

    add_subdirectory(deeplearning/yolov4_gpu)

    add_subdirectory(deeplearning/unet_gpu)

    add_subdirectory(deeplearning/deeplabv3_gpu)

    add_subdirectory(deeplearning/yolov5s_gpu)

    add_subdirectory(deeplearning/yolov5m_gpu)
    
    if(BUILD_USE_CUDNN MATCHES ON)

        add_subdirectory(deeplearning/mobilenetv2_yolov3_lite_gpu)
        
        add_subdirectory(deeplearning/mobilenetv2_yolov3_nano_gpu)

        add_subdirectory(deeplearning/yoloface100k_gpu)

        add_subdirectory(deeplearning/yoloface500k_gpu)

        add_subdirectory(deeplearning/unet_gpu_fp16)

        add_subdirectory(deeplearning/yolov3_gpu_fp16)

        add_subdirectory(deeplearning/yolov3tiny_gpu_fp16)

        add_subdirectory(deeplearning/yolov4_gpu_fp16)
    endif()
endif()

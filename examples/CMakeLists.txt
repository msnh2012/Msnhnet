add_subdirectory(lenet5)

add_subdirectory(classify)

add_subdirectory(yolov3)

add_subdirectory(yolov3tiny)

add_subdirectory(yolov3tiny_video)

add_subdirectory(yolov4)

add_subdirectory(unet)

add_subdirectory(fcns)

if(BUILD_USE_GPU MATCHES ON)
    add_subdirectory(classify_gpu)

    add_subdirectory(yolov3_gpu)

    add_subdirectory(yolov3tiny_gpu)

    add_subdirectory(yolov3tiny_video_gpu)

    add_subdirectory(yolov4_gpu)

    add_subdirectory(unet_gpu)

    if(BUILD_USE_CUDNN MATCHES ON)
        add_subdirectory(unet_gpu_fp16)
        add_subdirectory(yolov3_gpu_fp16)
        add_subdirectory(yolov3tiny_gpu_fp16)
        add_subdirectory(yolov4_gpu_fp16)
    endif()
endif()

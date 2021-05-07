file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(mobilenetv2_yolov3_lite_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(mobilenetv2_yolov3_lite_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(mobilenetv2_yolov3_lite_gpu Msnhnet)

install(TARGETS mobilenetv2_yolov3_lite_gpu
        RUNTIME DESTINATION bin)

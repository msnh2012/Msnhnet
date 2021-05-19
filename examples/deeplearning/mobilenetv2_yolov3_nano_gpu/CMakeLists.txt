file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(mobilenetv2_yolov3_nano_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(mobilenetv2_yolov3_nano_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(mobilenetv2_yolov3_nano_gpu Msnhnet)

install(TARGETS mobilenetv2_yolov3_nano_gpu
        RUNTIME DESTINATION bin)

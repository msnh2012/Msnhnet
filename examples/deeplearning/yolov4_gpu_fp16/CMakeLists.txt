file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov4_gpu_fp16 ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov4_gpu_fp16
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov4_gpu_fp16 Msnhnet)

install(TARGETS yolov4_gpu_fp16
        RUNTIME DESTINATION bin)

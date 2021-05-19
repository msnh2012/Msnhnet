file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(unet_gpu_fp16 ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(unet_gpu_fp16
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(unet_gpu_fp16 Msnhnet)

install(TARGETS unet_gpu_fp16
        RUNTIME DESTINATION bin)

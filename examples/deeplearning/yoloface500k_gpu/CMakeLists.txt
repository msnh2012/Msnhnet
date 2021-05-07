file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yoloface500k_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yoloface500k_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yoloface500k_gpu Msnhnet)

install(TARGETS yoloface500k_gpu
        RUNTIME DESTINATION bin)

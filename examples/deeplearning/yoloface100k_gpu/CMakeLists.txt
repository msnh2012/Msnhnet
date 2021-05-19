file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yoloface100k_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yoloface100k_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yoloface100k_gpu Msnhnet)

install(TARGETS yoloface100k_gpu
        RUNTIME DESTINATION bin)

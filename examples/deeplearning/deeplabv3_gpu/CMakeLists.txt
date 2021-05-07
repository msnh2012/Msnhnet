file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(deeplabv3_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(deeplabv3_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(deeplabv3_gpu Msnhnet)

install(TARGETS deeplabv3_gpu
        RUNTIME DESTINATION bin)

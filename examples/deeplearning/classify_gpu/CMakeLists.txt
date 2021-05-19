file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(classify_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(classify_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(classify_gpu Msnhnet)

install(TARGETS classify_gpu
        RUNTIME DESTINATION bin)

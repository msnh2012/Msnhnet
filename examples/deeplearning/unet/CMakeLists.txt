file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(unet ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(unet
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(unet Msnhnet)

install(TARGETS unet
        RUNTIME DESTINATION bin)

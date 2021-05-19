file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(lenet5 ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(lenet5
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(lenet5 Msnhnet)

install(TARGETS lenet5
        RUNTIME DESTINATION bin)

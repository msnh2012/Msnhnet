file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(landmark106 ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(landmark106
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(landmark106 Msnhnet)

install(TARGETS landmark106
        RUNTIME DESTINATION bin)

file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(classify ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(classify
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(classify Msnhnet)

install(TARGETS classify
        RUNTIME DESTINATION bin)

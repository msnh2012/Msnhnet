file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(fcns ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(fcns
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(fcns Msnhnet)

install(TARGETS fcns
        RUNTIME DESTINATION bin)

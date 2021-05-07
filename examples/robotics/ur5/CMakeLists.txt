file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(ur5 ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(ur5 PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(ur5 Msnhnet)

install(TARGETS ur5 RUNTIME DESTINATION bin)

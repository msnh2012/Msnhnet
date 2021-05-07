file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(puma560 ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(puma560 PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(puma560 Msnhnet)

install(TARGETS puma560 RUNTIME DESTINATION bin)

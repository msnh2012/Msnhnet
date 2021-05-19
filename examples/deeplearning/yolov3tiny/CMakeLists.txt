file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov3tiny ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov3tiny
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov3tiny Msnhnet)

install(TARGETS yolov3tiny
        RUNTIME DESTINATION bin)

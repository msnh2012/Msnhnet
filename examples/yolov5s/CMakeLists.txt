file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov5s ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov5s
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov5s Msnhnet)

install(TARGETS yolov5s
        RUNTIME DESTINATION bin)

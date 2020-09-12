file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov5m ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov5m
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov5m Msnhnet)

install(TARGETS yolov5m
        RUNTIME DESTINATION bin)

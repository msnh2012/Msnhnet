file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov5s_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov5s_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov5s_gpu Msnhnet)

install(TARGETS yolov5s_gpu
        RUNTIME DESTINATION bin)

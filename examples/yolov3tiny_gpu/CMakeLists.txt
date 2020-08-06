file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov3tiny_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov3tiny_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov3tiny_gpu Msnhnet)

install(TARGETS yolov3tiny_gpu
        RUNTIME DESTINATION bin)

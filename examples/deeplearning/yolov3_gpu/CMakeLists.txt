file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov3_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov3_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov3_gpu Msnhnet)

install(TARGETS yolov3_gpu
        RUNTIME DESTINATION bin)

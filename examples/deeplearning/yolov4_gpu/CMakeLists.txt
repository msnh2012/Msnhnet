file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov4_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov4_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov4_gpu Msnhnet)

install(TARGETS yolov4_gpu
        RUNTIME DESTINATION bin)

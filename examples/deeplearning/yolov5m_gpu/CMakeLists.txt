file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov5m_gpu ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov5m_gpu
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov5m_gpu Msnhnet)

install(TARGETS yolov5m_gpu
        RUNTIME DESTINATION bin)

file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov3 ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov3
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov3 Msnhnet)

install(TARGETS yolov3
        RUNTIME DESTINATION bin)

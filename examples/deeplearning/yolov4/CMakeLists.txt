file(GLOB_RECURSE CPPS  ./*.cpp )

add_executable(yolov4 ${CPPS})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(yolov4
                               PRIVATE USE_SHARED_MSNHNET)
endif()

target_link_libraries(yolov4 Msnhnet)

install(TARGETS yolov4
        RUNTIME DESTINATION bin)

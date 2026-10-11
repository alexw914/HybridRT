if(NOT TARGET opencv_imgcodecs)
    message(STATUS "Examples skipped: OpenCV imgcodecs not found")
    return()
endif()

add_executable(hybridrt_example_image_model image_model.cpp)
target_link_libraries(hybridrt_example_image_model
    PRIVATE hrt::hybridrt kitzoo::core kitzoo::log opencv_imgcodecs)
hybridrt_apply_warnings(hybridrt_example_image_model)

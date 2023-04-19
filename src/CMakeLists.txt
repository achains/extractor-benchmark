#####################################
# CAPE Image Processor
#####################################
add_executable(deplex_image_processor
    deplex/image_processor.cpp)

target_link_libraries(deplex_image_processor PRIVATE deplex)
target_compile_options(deplex_image_processor PRIVATE -O3)
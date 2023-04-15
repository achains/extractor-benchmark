#####################################
# Config variables
#####################################


#####################################
# CAPE Benchmark
#####################################
add_executable(cape_benchmark cape_benchmark.cpp)

target_link_libraries(cape_benchmark PRIVATE cape)
target_compile_options(cape_benchmark PRIVATE -O3)
target_include_directories(cape_benchmark SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
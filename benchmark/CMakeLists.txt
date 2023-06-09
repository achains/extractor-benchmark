#####################################
# Config variables
#####################################

set(DATA_DIR ${CMAKE_SOURCE_DIR}/data)

if (NOT EXISTS ${DATASET_NAME})
    set(DATASET_NAME "${DATA_DIR}/tum/long_office")
endif()

if (NOT EXISTS ${OUTPUT_DIR})
    set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/output)
    file(MAKE_DIRECTORY ${OUTPUT_DIR})
endif()

option(BENCHMARK_VERBOSE OFF)
if (${BENCHMARK_VERBOSE})
    add_compile_definitions(BENCHMARK_VERBOSE)
endif()

set(CAPE_CAMERA_INTRINSICS "${DATASET_NAME}/cape.xml")
set(PEAC_CAMERA_INTRINSICS "${DATASET_NAME}/intrinsics.K")
set(DEPLEX_CAMERA_INTRINSICS "${DATASET_NAME}/intrinsics.K")

set(CAPE_CONFIG "${DATASET_NAME}/cape.ini")
set(PEAC_CONFIG "${DATASET_NAME}/peac.ini")
set(DEPLEX_CONFIG "${DATASET_NAME}/deplex.ini")

configure_file(config.h.in config.h)

#####################################
# CAPE Benchmark
#####################################
add_executable(cape_benchmark cape_benchmark.cpp)

target_link_libraries(cape_benchmark PRIVATE cape)
target_compile_options(cape_benchmark PRIVATE -O3)
target_include_directories(cape_benchmark SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

#####################################
# PEAC Benchmark
#####################################
add_executable(peac_benchmark peac_benchmark.cpp)

target_link_libraries(peac_benchmark PRIVATE peac)
# Needed for image utils
target_link_libraries(peac_benchmark PRIVATE deplex)
target_compile_options(peac_benchmark PRIVATE -O3 -fpermissive)
target_include_directories(peac_benchmark SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

#####################################
# deplex Benchmark
#####################################
add_executable(deplex_benchmark deplex_benchmark.cpp)

target_link_libraries(deplex_benchmark PRIVATE deplex)
target_compile_options(deplex_benchmark PRIVATE -O3)
target_include_directories(deplex_benchmark SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(cape
    CAPE.cpp
    CylinderSeg.cpp
    Histogram.cpp
    Params.cpp
    PlaneSeg.cpp)

if (not ${PROJECT_COMPILE_OPTIMIZATION})
    target_compile_options(cape PRIVATE -O3)
endif()

target_include_directories(cape PUBLIC .)

target_link_libraries(cape PUBLIC Eigen3::Eigen)
target_link_libraries(cape PUBLIC ${OpenCV_LIBS})
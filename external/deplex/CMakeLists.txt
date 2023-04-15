find_package(deplex QUIET)

include(FetchContent)
FetchContent_Declare(
    deplex
    GIT_REPOSITORY https://github.com/prime-slam/deplex.git
    GIT_TAG v1.0.5
)

FetchContent_MakeAvailable(deplex)
find_package(deplex QUIET)

include(FetchContent)
FetchContent_Declare(
    deplex
    GIT_REPOSITORY https://github.com/prime-slam/deplex.git
    GIT_TAG b44c768f3b3dcf07eed3f4ad4dee0ca88f4a409e
)

FetchContent_MakeAvailable(deplex)
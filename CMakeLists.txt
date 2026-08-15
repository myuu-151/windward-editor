cmake_minimum_required(VERSION 3.24)
project(terrain_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# SDL3, pinned release, built statically into the exe (no DLL juggling)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    SDL3
    URL https://github.com/libsdl-org/SDL/releases/download/release-3.4.14/SDL3-3.4.14.zip
)
FetchContent_MakeAvailable(SDL3)

add_executable(terrain
    src/main.cpp
    src/gl_loader.cpp
)
target_compile_definitions(terrain PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(terrain PRIVATE SDL3::SDL3-static opengl32)

if(MSVC)
    target_compile_options(terrain PRIVATE /W4)
    set_target_properties(terrain PROPERTIES WIN32_EXECUTABLE $<CONFIG:Release>)
endif()

cmake_minimum_required(VERSION 3.10)
project(Denoiser)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DNOMINMAX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(OptiX80)
# Add CUDA
find_package(CUDAToolkit 12.0 REQUIRED)
set(OIIO_DIR ${PROJECT_SOURCE_DIR}/contrib/OpenImageIO)
find_package(OpenImageIO REQUIRED)

# Include directories
include_directories(${OPTIX80_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${OIIO_INCLUDE_DIR})

# Source files
set(SOURCES src/main.cpp)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OIIO_LIBRARY} CUDA::cudart)

# Copy DLLs to output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${PROJECT_SOURCE_DIR}/contrib/OpenImageIO/bin
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

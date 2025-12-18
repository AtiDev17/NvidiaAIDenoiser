cmake_minimum_required(VERSION 3.10)
project(Denoiser)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-DNOMINMAX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(OptiX REQUIRED)
# Add CUDA
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)

# Include directories
include_directories(${OPTIX_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(SOURCES 
    src/main.cpp
    src/CudaCheck.h
    src/CudaBuffer.h
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    OpenImageIO::OpenImageIO 
    CUDA::cudart
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(FILES README.md LICENSE DESTINATION .)
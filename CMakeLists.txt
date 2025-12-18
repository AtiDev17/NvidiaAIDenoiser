cmake_minimum_required(VERSION 3.21)
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

# Attempt to find the runtime DLL via the target or standard paths
get_target_property(CUDART_LIB_LOCATION CUDA::cudart IMPORTED_LOCATION)
get_filename_component(CUDA_ROOT_DIR "${CUDAToolkit_BIN_DIR}/.." ABSOLUTE)

# The user pointed out that on this system, binaries are in bin/x64
file(GLOB CUDA_RUNTIME_DLL "${CUDA_ROOT_DIR}/bin/x64/cudart64*.dll")

if (NOT CUDA_RUNTIME_DLL)
    # Fallback: try standard bin location
    file(GLOB CUDA_RUNTIME_DLL "${CUDA_ROOT_DIR}/bin/cudart64*.dll")
endif()

if (NOT CUDA_RUNTIME_DLL)
    # Fallback: try relative to lib location
    get_filename_component(CUDART_LIB_DIR "${CUDART_LIB_LOCATION}" DIRECTORY)
    file(GLOB CUDA_RUNTIME_DLL "${CUDART_LIB_DIR}/../bin/cudart64*.dll")
endif()

message(STATUS "CUDAToolkit_BIN_DIR: ${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDA::cudart location: ${CUDART_LIB_LOCATION}")
message(STATUS "Resolved CUDA Runtime DLL: ${CUDA_RUNTIME_DLL}")

# Post-build: Copy DLLs to the build directory next to the executable
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${PROJECT_NAME}> $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> ${CUDA_RUNTIME_DLL}
    COMMAND_EXPAND_LISTS
)

# Install: Copy executable and DLLs to 'bin'
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(FILES $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> ${CUDA_RUNTIME_DLL} DESTINATION bin)
install(FILES README.md LICENSE DESTINATION .)
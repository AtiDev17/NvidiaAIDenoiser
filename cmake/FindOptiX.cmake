# Looks for the environment variable:
# OPTIX_PATH or OPTIX_ROOT

# Sets the variables :
# OPTIX_INCLUDE_DIR
# OptiX_FOUND

set(OPTIX_PATH $ENV{OPTIX_PATH})
if("${OPTIX_PATH}" STREQUAL "")
    set(OPTIX_PATH $ENV{OPTIX_ROOT})
endif()

if("${OPTIX_PATH}" STREQUAL "")
    if(WIN32)
        # Try finding it inside the default installation directory under Windows first.
        # Check for OptiX 9 first
        if (EXISTS "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0")
            set(OPTIX_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0")
        elseif (EXISTS "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
            set(OPTIX_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
        elseif (EXISTS "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
            set(OPTIX_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
        endif()
    else()
        # Adjust this if the OptiX SDK installation is in a different location.
        if (EXISTS "~/NVIDIA-OptiX-SDK-9.1.0-linux64")
             set(OPTIX_PATH "~/NVIDIA-OptiX-SDK-9.1.0-linux64")
        elseif (EXISTS "~/NVIDIA-OptiX-SDK-9.0.0-linux64")
             set(OPTIX_PATH "~/NVIDIA-OptiX-SDK-9.0.0-linux64")
        endif()
    endif()
endif()

find_path(OPTIX_INCLUDE_DIR optix_host.h ${OPTIX_PATH}/include)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX DEFAULT_MSG OPTIX_INCLUDE_DIR)

mark_as_advanced(OPTIX_INCLUDE_DIR)

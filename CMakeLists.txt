cmake_minimum_required(VERSION 3.14...3.25)

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT (CMAKE_BUILD_TYPE OR DEFINED ENV{CMAKE_BUILD_TYPE}))
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Release default")
endif()

project(PETALFLOW LANGUAGES C)

option(TESTS "Build tests instead of shared library" ON)
option(LOGGING "Enable logging into stdout" ON)
set(LOGGER_LEVEL "1" CACHE STRING "Logging level (0 - Debug, 1 - Info, 2 - Warning, 3 - Error, 255 - No logging)")
option(LOGGER_DISABLE_TIME "Disable date and time logging" OFF)
set(LOGGER_TIME_FORMAT "[%Y-%m-%d %H:%M:%S]" CACHE STRING "Logging time formatter")
option(LOGGER_DISABLE_LEVEL "Disable logging level" OFF)
option(LOGGER_LEVEL_FIXED "Enable fixed-width logging level printing" OFF)
set(LOGGER_LEVEL_FIXED_FORMAT "[%-7s]" CACHE STRING "Fixed-width level formatter")
option(LOGGER_DISABLE_TAG "Disable logging tag" OFF)

include(ExternalProject)
include(GNUInstallDirs)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}" CACHE PATH "default install path" FORCE)
endif()

# Source files
file(GLOB PETALFLOW_SRC CONFIGURE_DEPENDS "${PETALFLOW_SOURCE_DIR}/include/*.h" "${PETALFLOW_SOURCE_DIR}/src/*.c")

# Build tests
if(TESTS)
	set(fw_url https://github.com/F33RNI/PetalFlow.git)
	set(fw_tag main)
	set(fw_args
	-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
	-DCMAKE_BUILD_TYPE=Release
	-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
	)

    # Tests executable (main.c)
    add_executable(petalflow_tests "${PETALFLOW_SOURCE_DIR}/test/main.c" ${PETALFLOW_SRC})

    # Link header files
    target_include_directories(petalflow_tests PRIVATE "${PETALFLOW_SOURCE_DIR}/include")

    # Link math library
    target_link_libraries(petalflow_tests m)

    # Logger definitions
    if(LOGGING)
        target_compile_definitions(petalflow_tests PRIVATE LOGGING)
        target_compile_definitions(petalflow_tests PRIVATE LOGGER_LEVEL=${LOGGER_LEVEL})
        if(LOGGER_DISABLE_TIME)
            target_compile_definitions(petalflow_tests PRIVATE LOGGER_DISABLE_TIME)
        endif()
        target_compile_definitions(petalflow_tests PRIVATE LOGGER_TIME_FORMAT="${LOGGER_TIME_FORMAT}")
        if(LOGGER_DISABLE_LEVEL)
            target_compile_definitions(petalflow_tests PRIVATE LOGGER_DISABLE_LEVEL)
        endif()
        if(LOGGER_LEVEL_FIXED)
            target_compile_definitions(petalflow_tests PRIVATE LOGGER_LEVEL_FIXED)
        endif()
        target_compile_definitions(petalflow_tests PRIVATE LOGGER_LEVEL_FIXED_FORMAT="${LOGGER_LEVEL_FIXED_FORMAT}")
        if(LOGGER_DISABLE_TAG)
            target_compile_definitions(petalflow_tests PRIVATE LOGGER_DISABLE_TAG)
        endif()
    endif()

# Build shared library
else()
    # TODO
endif()

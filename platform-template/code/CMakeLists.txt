cmake_minimum_required(VERSION 3.16.0)
project(mish_op_prj)
find_package(ASC REQUIRED)
if(NOT DEFINED CANNJUDGE_COMPILE_JOBS)
    set(CANNJUDGE_COMPILE_JOBS 4)
endif()
set(CANN_HOST_ARCH "${CMAKE_SYSTEM_PROCESSOR}-linux")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ARCH32_COMPUTE_UNITS ascend910b ascend910_93)
set(ARCH35_COMPUTE_UNITS ascend950)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${ARCH32_COMPUTE_UNITS} ${ARCH35_COMPUTE_UNITS})
endif()
set(package_name mish_custom)

 npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()



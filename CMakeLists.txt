cmake_minimum_required(VERSION 3.7)
project(simtod)

set (TOP_DIR ${CMAKE_CURRENT_LIST_DIR})
set (CMAKE_CXX_FLAGS "-std=c++11 -march=native -Og -g" CACHE STRING "Set C++ Compiler Flags" FORCE)
set (LDFLAG "")


set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${TOP_DIR}" CACHE PATH "${TOP_DIR}" FORCE)
endif()

if(${CMAKE_CXX_COMPILER_ID} MATCHES "[a-zA-Z]*Clang")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set (LDFLAG "-lomp")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -flto")
else()
    message(FATAL_ERROR "Your compiler is not support for now.")
endif()

find_package(Python3 COMPONENTS Development NumPy)

add_subdirectory(${TOP_DIR}/beam_cxx)

add_library(${CMAKE_PROJECT_NAME} SHARED
    ./src/PyMapMaking.cxx
    ./src/simtodmodule.cxx
    ./src/modulefuncs.cxx
    ./src/helper.cpp
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME "_simtod")
target_link_libraries( ${CMAKE_PROJECT_NAME}
    PRIVATE
    Python3::Python
    Python3::NumPy
    mapmaking
    convolve
)

include_directories( ${CMAKE_PROJECT_NAME}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/beam_cxx/include
)


install (TARGETS ${CMAKE_PROJECT_NAME} LIBRARY DESTINATION . )

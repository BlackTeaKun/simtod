cmake_minimum_required(VERSION 3.7)
project(convolve)

set (CMAKE_CXX_FLAGS "-std=c++11 -march=native -Og -g" CACHE STRING "Set C++ Compiler Flags" FORCE)
set (LDFLAG "")

if(${CMAKE_CXX_COMPILER_ID} MATCHES "[a-zA-Z]*Clang")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set (LDFLAG "-lomp")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -flto")
else()
    message(FATAL_ERROR "Your compiler is not support for now.")
endif()


find_package(PkgConfig REQUIRED)
pkg_search_module(HEALPIX REQUIRED healpix_cxx)
pkg_search_module(OPENBLAS openblas)

set (LIBDIR "${HEALPIX_LIBDIR}")
if(OPENBLAS_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_BLAS")
    set(LIBDIR "${LIBDIR} ${OPENBLAS_LIBDIR}")
endif()

message(${CMAKE_CXX_FLAGS})

add_library(convolve SHARED
    src/Beam.cxx
    src/convolve.cxx
)
add_library(mapmaking SHARED
    src/MapMaking.cxx
)
link_directories ( convolve
    PRIVATE
    ${LIBDIR}
)
target_link_libraries( convolve
    PRIVATE
    ${HEALPIX_LDFLAGS}
    ${OPENBLAS_LDFLAGS}
    ${LDFLAG}
)
target_link_libraries( mapmaking
    PRIVATE
    ${HEALPIX_LDFLAGS}
    ${OPENBLAS_LDFLAGS}
    ${LDFLAG}
)

include_directories( convolve
    PRIVATE
    ${HEALPIX_INCLUDEDIR}
    ${PROJECT_SOURCE_DIR}/include
)


install (TARGETS convolve mapmaking LIBRARY DESTINATION lib)


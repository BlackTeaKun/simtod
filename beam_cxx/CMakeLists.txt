cmake_minimum_required(VERSION 3.6)
project(convolve)

set (CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -march=native -O2 -DEIGEN_USE_BLAS" CACHE STRING "Set C++ Compiler Flags" FORCE)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
find_package(PkgConfig REQUIRED)
pkg_search_module(HEALPIX REQUIRED healpix_cxx)
pkg_search_module(OPENBLAS REQUIRED openblas)

add_library(convolve SHARED
    src/Beam.cxx
    src/convolve.cxx
    src/MapMaking.cxx
)
target_link_directories ( convolve
    PRIVATE
    ${HEALPIX_LIBDIR}
    ${OPENBLAS_LIBDIR}
)
target_link_libraries( convolve
    PRIVATE
    ${HEALPIX_LDFLAGS}
    ${OPENBLAS_LDFLAGS}
)

target_include_directories( convolve
    PRIVATE
    ${HEALPIX_INCLUDEDIR}
    ${PROJECT_SOURCE_DIR}/include
)


install (TARGETS convolve LIBRARY DESTINATION lib)

# Find OpenBLAS
#
#  OpenBLAS_FOUND       - True if OpenBLAS is found
#  CBLAS_INCLUDE_DIR - OpenBLAS include directory
#  CBLAS_LIBRARIES   - OpenBLAS library path

find_path(OpenBLAS_INCLUDE_DIR NAMES cblas.h)
find_library(OpenBLAS_LIBRARIES NAMES openblas)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARIES)

if(OpenBLAS_FOUND)
  message(STATUS "Found OpenBLAS (include: ${OpenBLAS_INCLUDE_DIR}, library: ${OpenBLAS_LIBRARIES})")
  mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARIES)
endif()
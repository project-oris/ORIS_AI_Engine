# Find Glog
#
#  Glog_FOUND       - True if GLog is found
#  Glog_INCLUDE_DIR - GLog include directory
#  GLog_LIBRARIES   - GLog library path

find_path(GLog_INCLUDE_DIR NAMES glog/logging.h)
find_library(GLog_LIBRARIES NAMES glog PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Glog DEFAULT_MSG GLog_INCLUDE_DIR GLog_LIBRARIES)

if(GLOG_FOUND)
  message(STATUS "Found glog (include: ${GLog_INCLUDE_DIR}, library: ${GLog_LIBRARIES})")
  mark_as_advanced(GLog_INCLUDE_DIR GLog_LIBRARIES)
endif()

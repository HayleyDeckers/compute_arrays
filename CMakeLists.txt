cmake_minimum_required (VERSION 2.6)
project (compute_arrays)
# The version number.
set (compute_arrays_VERSION_MAJOR 1)
set (compute_arrays_VERSION_MINOR 0)

#install(TARGETS compute_arrays)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include
)

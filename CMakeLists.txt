cmake_minimum_required (VERSION 2.8)

# projectname is the same as the main-executable
project(logistic_regression_ps)

add_definitions('-g')
add_definitions('-Wall')
add_definitions('-std=c++11')

include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/ps-lite/include"
  "${PROJECT_BINARY_DIR}"
)
message(${PROJECT_SOURCE_DIR})

link_directories(
  "${PROJECT_BINARY_DIR}"
  "${PROJECT_BINARY_DIR}/ps-lite"
)

add_subdirectory(ps-lite)
add_subdirectory(src)


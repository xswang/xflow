set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/src/io)

add_library(io STATIC load_data_from_disk.cc)
target_link_libraries(io)

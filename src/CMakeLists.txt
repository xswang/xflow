set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/src)

set(STA_DEPS pslite protobuf zmq)
add_executable(xflow_lr main.cc io/load_data_from_local.cc)
target_link_libraries(xflow_lr ${STA_DEPS})



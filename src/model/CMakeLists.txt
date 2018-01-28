set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/src)

set(STA_DEPS io pslite protobuf zmq)
add_executable(xflow_lr main.cc lr/lr_worker.cc fm/fm_worker.cc mvm/mvm_worker.cc)
target_link_libraries(xflow_lr ${STA_DEPS})

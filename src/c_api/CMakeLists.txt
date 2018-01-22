set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/src/c_api)

#set(STA_DEPS xflow_lr)
#add_library(xflow_api STATIC c_api.cc)
#target_link_libraries(xflow_api ${STA_DEPS})

add_library(xflow_api_shared SHARED c_api.cc)
target_link_libraries(xflow_api_shared pslite protobuf zmq)
set_target_properties(xflow_api_shared PROPERTIES OUTPUT_NAME "xflow_api")

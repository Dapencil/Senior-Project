cmake_minimum_required(VERSION 3.6)
project(ckks_relu_square)

add_executable(${PROJECT_NAME} 
main.cpp 
bias_relu_0.cpp 
weight_relu_0.cpp
bias_relu_2.cpp 
weight_relu_2.cpp
bias_relu_4.cpp 
weight_relu_4.cpp
bias_relu_6.cpp 
weight_relu_6.cpp)
find_package(Protobuf REQUIRED)
find_package(SEAL 4.1 REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} SEAL::seal protobuf::libprotobuf PROJECT_LIB)
add_subdirectory(lib)
cmake_minimum_required(VERSION 3.6)
project(mnist_ckks_16384)

add_executable(${PROJECT_NAME} 
main.cpp 
bias_square_0.cpp 
weight_square_0.cpp
bias_square_2.cpp 
weight_square_2.cpp
bias_square_4.cpp 
weight_square_4.cpp
bias_square_6.cpp 
weight_square_6.cpp)
find_package(Protobuf REQUIRED)
find_package(SEAL 4.1 REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} SEAL::seal protobuf::libprotobuf PROJECT_LIB)
add_subdirectory(lib)
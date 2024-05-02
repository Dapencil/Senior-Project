add_library(PROJECT_LIB 
    Util.cpp
    senior.pb.cc
)

target_include_directories(PROJECT_LIB INTERFACE header)
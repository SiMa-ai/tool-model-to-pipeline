cmake_minimum_required(VERSION 3.0)
project(YourProjectName)

# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(jsoncpp REQUIRED)

# Add executable
add_executable(run_inference main.cpp)

# Link OpenCV libraries
target_link_libraries(run_inference ${OpenCV_LIBS} simacpplib JsonCpp::JsonCpp)

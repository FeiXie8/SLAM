cmake_minimum_required(VERSION 2.4.6)
project(usesophus)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV 4.5 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)
#find_package(G2O REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} 
                    ${Sophus_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Ceres_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    
                    )
add_executable(useSophus ${PROJECT_SOURCE_DIR}/src/useSophus.cpp)
target_link_libraries(useSophus ${Sophus_LIBS} fmt)

add_executable(antagri ${PROJECT_SOURCE_DIR}/src/antagri.cpp)
target_link_libraries(antagri )

add_executable(antagriCV ${PROJECT_SOURCE_DIR}/src/antgriWithCV.cpp)
target_link_libraries(antagriCV ${OpenCV_LIBS})

add_executable(gaussNewton ${PROJECT_SOURCE_DIR}/src/gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting ${PROJECT_SOURCE_DIR}/src/ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(g2oCurveFitting ${PROJECT_SOURCE_DIR}/src/g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} g2o_stuff g2o_core)

add_executable(orb_cv ${PROJECT_SOURCE_DIR}/src/orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})
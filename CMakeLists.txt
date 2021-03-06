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
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)
#find_package(G2O REQUIRED)

add_definitions(${PCL_DEFINITIONS})

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )


include_directories(${EIGEN3_INCLUDE_DIR} 
                    ${Sophus_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Ceres_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS}
                    ./include
                    )
add_executable(useSophus ${PROJECT_SOURCE_DIR}/src/useSophus.cpp)
target_link_libraries(useSophus ${Sophus_LIBS} fmt)

add_executable(antagri ${PROJECT_SOURCE_DIR}/src/antagri.cpp)
target_link_libraries(antagri )

add_executable(acs ${PROJECT_SOURCE_DIR}/src/acs.cpp)
target_link_libraries(acs )

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

add_executable(pose_estimation_2d2d ${PROJECT_SOURCE_DIR}/src/pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d ${PROJECT_SOURCE_DIR}/src/pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS}
                                           ${Sophus_LIBS}
                                           fmt
                                           g2o_stuff g2o_core
                                           )

add_executable(triangulation ${PROJECT_SOURCE_DIR}/src/triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

add_executable(derect_method ${PROJECT_SOURCE_DIR}/src/derect_method.cpp)
target_link_libraries(derect_method ${OpenCV_LIBS} 
                                    ${Sophus_LIBS} 
                                    fmt 
                                    ${Pangolin_LIBRARIES})

add_executable(ba_ceres ${PROJECT_SOURCE_DIR}/src/ba_ceres.cpp ${PROJECT_SOURCE_DIR}/src/common.cpp)
target_link_libraries(ba_ceres ${CERES_LIBRARIES})

add_executable(feature_training ${PROJECT_SOURCE_DIR}/src/feature_training.cpp)
target_link_libraries(feature_training ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(pointcloud ${PROJECT_SOURCE_DIR}/src/pointcloud.cpp)
target_link_libraries(pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES})
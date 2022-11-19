find_package(Arrow REQUIRED)
find_package(vineyard REQUIRED)

add_graphlab_executable(vineyard-test hello_world.cpp)
target_include_directories(vineyard-test PRIVATE ${Arrow_INCLUDE_DIR} ${VINEYARD_INCLUDE_DIR})
target_link_libraries(vineyard-test PRIVATE arrow_shared ${VINEYARD_LIBRARIES})

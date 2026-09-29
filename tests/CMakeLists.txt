find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(ckl-memory-planner-test Core/MemoryPlannerTest.cpp)
target_link_libraries(ckl-memory-planner-test PRIVATE CKLMemoryPlanner)
target_compile_features(ckl-memory-planner-test PRIVATE cxx_std_17)
target_compile_options(ckl-memory-planner-test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME ckl-memory-planner COMMAND ckl-memory-planner-test)

add_test(
  NAME ckl-python-compiler-utils
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${PROJECT_SOURCE_DIR}/python
    ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/Python -p "test_*.py"
)

if(CKL_ENABLE_MLIR)
  add_test(
    NAME ckl-mlir-analysis
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MLIR/test_semantics.py
        $<TARGET_FILE:ckl-opt> ${CMAKE_CURRENT_SOURCE_DIR}/MLIR
  )
endif()

if(CKL_ENABLE_CUDA_RUNTIME)
  add_subdirectory(Runtime)
endif()

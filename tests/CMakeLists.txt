find_package(Python3 REQUIRED COMPONENTS Interpreter)

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

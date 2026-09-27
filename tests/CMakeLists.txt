find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME ckl-python-compiler-utils
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${PROJECT_SOURCE_DIR}/python
    ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR}/Python -p "test_*.py"
)

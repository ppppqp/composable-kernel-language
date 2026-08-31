find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME ckl-python-unit
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${PROJECT_SOURCE_DIR}/python
    ${Python3_EXECUTABLE} -m unittest discover
      -s ${CMAKE_CURRENT_SOURCE_DIR} -p test_frontend.py
)

add_test(
  NAME ckl-python-mlir-roundtrip
  COMMAND ${CMAKE_COMMAND}
    -DCKL_OPT=$<TARGET_FILE:ckl-opt>
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DPYTHON_DIR=${PROJECT_SOURCE_DIR}/python
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/emit_copy.py
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_frontend.cmake
)

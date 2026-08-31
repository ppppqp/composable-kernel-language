set(CKL_PYTHON_GEN_DIR "${PROJECT_BINARY_DIR}/python/ckl/dialects")
set(CKL_PYTHON_OPS "${CKL_PYTHON_GEN_DIR}/_ckl_ops_gen.py")
set(CKL_PYTHON_TABLEGEN_INCLUDES
  -I "${PROJECT_SOURCE_DIR}/include/ckl/Dialect/CKL/IR"
  -I "${PROJECT_SOURCE_DIR}/include"
)
foreach(include_dir IN LISTS MLIR_INCLUDE_DIRS)
  list(APPEND CKL_PYTHON_TABLEGEN_INCLUDES -I "${include_dir}")
endforeach()

add_custom_command(
  OUTPUT "${CKL_PYTHON_OPS}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CKL_PYTHON_GEN_DIR}"
  COMMAND ${MLIR_TABLEGEN_EXE}
    -gen-python-op-bindings
    -bind-dialect=ckl
    ${CKL_PYTHON_TABLEGEN_INCLUDES}
    "${PROJECT_SOURCE_DIR}/include/ckl/Dialect/CKL/IR/CKLOps.td"
    -o "${CKL_PYTHON_OPS}"
  DEPENDS
    "${PROJECT_SOURCE_DIR}/include/ckl/Dialect/CKL/IR/CKLDialect.td"
    "${PROJECT_SOURCE_DIR}/include/ckl/Dialect/CKL/IR/CKLOps.td"
    "${PROJECT_SOURCE_DIR}/include/ckl/Dialect/CKL/IR/CKLTypes.td"
  VERBATIM
)

add_custom_target(CKLPythonBindings ALL DEPENDS "${CKL_PYTHON_OPS}")

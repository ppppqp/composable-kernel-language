add_executable(ckl-core-validation CoreValidation.cpp)
target_link_libraries(ckl-core-validation PRIVATE CKL::Core CKL::AMD)
target_compile_options(ckl-core-validation PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(ckl-core-validation PRIVATE CKL_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

add_test(NAME ckl-core-validation COMMAND ckl-core-validation)

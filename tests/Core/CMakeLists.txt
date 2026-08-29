add_executable(ckl-core-validation CoreValidation.cpp)
target_link_libraries(ckl-core-validation PRIVATE CKL::Core)
target_compile_options(ckl-core-validation PRIVATE -Wall -Wextra -Wpedantic)

add_test(NAME ckl-core-validation COMMAND ckl-core-validation)


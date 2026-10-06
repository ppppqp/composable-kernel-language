set(CKL_FLYDSL_STAGE1_BUNDLE_DIR "" CACHE PATH
  "Prepared FlyDSL Stage 1 bundle containing host.cpp.inc and device.hsaco")

if(CKL_FLYDSL_STAGE1_BUNDLE_DIR)
  if(NOT EXISTS "${CKL_FLYDSL_STAGE1_BUNDLE_DIR}/host.cpp.inc")
    message(FATAL_ERROR
      "CKL_FLYDSL_STAGE1_BUNDLE_DIR has no host.cpp.inc: ${CKL_FLYDSL_STAGE1_BUNDLE_DIR}")
  endif()
  add_library(ckl-flydsl-stage1-bridge SHARED bridge.cpp)
  target_include_directories(ckl-flydsl-stage1-bridge PRIVATE
    "${CKL_FLYDSL_STAGE1_BUNDLE_DIR}"
  )
  target_link_libraries(ckl-flydsl-stage1-bridge PRIVATE CKLHipRuntime)
  target_compile_features(ckl-flydsl-stage1-bridge PRIVATE cxx_std_17)
  target_compile_options(ckl-flydsl-stage1-bridge PRIVATE -Wall -Wextra -Wpedantic)
endif()

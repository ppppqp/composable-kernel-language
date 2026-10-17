#ifndef CKL_TRANSFORMS_CUDAKERNELS_H
#define CKL_TRANSFORMS_CUDAKERNELS_H

namespace mlir::ckl {
/// Register CUDA preparation and CKL kernel-to-GPU lowering.
void registerCKLCudaKernelPasses();
} // namespace mlir::ckl

#endif

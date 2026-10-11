// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/ascend/acl_common.cpp
// Description: Implements ACL error checks and process-wide Ascend setup.
// -----------------------------------------------------------------------------

#include "include/backend/ascend/acl_common.hpp"

#include <kitzoo/log/logger.hpp>

#include <acl/dvpp/hi_dvpp.h>

namespace hrt::backend {

AclRuntime::AclRuntime() : owned_(aclInit(nullptr) == ACL_SUCCESS) {}

AclRuntime::~AclRuntime() {
  if (owned_)
    aclFinalize();
}

// Not fatal: without the media system DVPP preprocessing is unavailable.
DvppRuntime::DvppRuntime() {
  const auto status = hi_mpi_sys_init();
  ready_ = status == ACL_SUCCESS;
  if (!ready_)
    KZ_LOG_ERROR("dvpp: hi_mpi_sys_init failed: 0x{:X}", static_cast<unsigned int>(status));
}

DvppRuntime::~DvppRuntime() {
  if (ready_)
    hi_mpi_sys_exit();
}

} // namespace hrt::backend

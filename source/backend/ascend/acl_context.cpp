// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/ascend/acl_context.cpp
// Description: Implements the Ascend device context and stream.
// -----------------------------------------------------------------------------

#include "include/backend/ascend/acl_context.hpp"

#include <kitzoo/log/logger.hpp>

#include "include/backend/ascend/acl_common.hpp"

namespace hrt::backend {

AclStream::AclStream() {
  aclrtStream stream = nullptr;
  KZ_CHECK_MSG(aclrtCreateStream(&stream) == ACL_SUCCESS, "acl: aclrtCreateStream failed: {}", aclGetRecentErrMsg());
  handle_ = stream;
}

AclStream::~AclStream() {
  if (handle_ != nullptr && aclrtDestroyStream(handle_) != ACL_SUCCESS)
    KZ_LOG_ERROR("acl: aclrtDestroyStream failed: {}", aclGetRecentErrMsg());
}

auto AclStream::synchronize() -> void {
  KZ_CHECK_RETURN(aclrtSynchronizeStream(handle_) == ACL_SUCCESS, , "acl: aclrtSynchronizeStream failed: {}",
                  aclGetRecentErrMsg());
}

AclContext::AclContext(int device_id) : core::Context(core::DeviceType::Ascend, device_id) {
  static_cast<void>(AclRuntime::instance());
  aclrtContext context = nullptr;
  // Not fatal: a null handle makes Context::create report the device unusable.
  KZ_CHECK_RETURN(aclrtSetDevice(device_id) == ACL_SUCCESS, , "acl: aclrtSetDevice({}) failed: {}", device_id,
                  aclGetRecentErrMsg());
  KZ_CHECK_RETURN(aclrtCreateContext(&context, device_id) == ACL_SUCCESS, , "acl: aclrtCreateContext failed: {}",
                  aclGetRecentErrMsg());
  handle_ = context;
}

AclContext::~AclContext() {
  if (handle_ != nullptr && aclrtDestroyContext(handle_) != ACL_SUCCESS)
    KZ_LOG_ERROR("acl: aclrtDestroyContext failed: {}", aclGetRecentErrMsg());
}

auto AclContext::set_current() -> void {
  aclrtContext current = nullptr;
  const auto status = aclrtGetCurrentContext(&current);
  if (status == ACL_SUCCESS && current == handle_)
    return;
  KZ_CHECK_RETURN(status == ACL_SUCCESS || status == ACL_ERROR_RT_CONTEXT_NULL, ,
                  "acl: aclrtGetCurrentContext failed: {}", aclGetRecentErrMsg());
  KZ_CHECK_RETURN(aclrtSetCurrentContext(handle_) == ACL_SUCCESS, , "acl: aclrtSetCurrentContext failed: {}",
                  aclGetRecentErrMsg());
}

auto AclContext::create_stream() -> UniquePtr<core::Stream> {
  set_current();
  return std::make_unique<AclStream>();
}

} // namespace hrt::backend

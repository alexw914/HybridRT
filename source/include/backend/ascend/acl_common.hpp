// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/ascend/acl_common.hpp
// Description: Declares ACL error checks and process-wide Ascend runtime setup.
// -----------------------------------------------------------------------------

#ifndef HRT_ASCEND_ACL_COMMON_HPP
#define HRT_ASCEND_ACL_COMMON_HPP

#include <hrt/core/context.hpp>

#include <kitzoo/core/macro.hpp>
#include <kitzoo/core/singleton.hpp>

#include <acl/acl.h>

namespace hrt::backend {

inline auto acl_stream(core::Stream* stream) -> aclrtStream {
  // aclrtStream is void*, so the handle needs no cast.
  return stream == nullptr ? nullptr : stream->handle();
}

// aclInit for the process, finalized at exit if this process initialized it.
class AclRuntime final : public kitzoo::core::Singleton<AclRuntime> {
public:
  ~AclRuntime();

private:
  friend class kitzoo::core::Singleton<AclRuntime>;

  AclRuntime();

  bool owned_ = false;
};

// hi_mpi_sys_init for the DVPP media system, once per process.
class DvppRuntime final : public kitzoo::core::Singleton<DvppRuntime> {
public:
  ~DvppRuntime();

  KZ_NODISCARD auto ready() const -> bool { return ready_; }

private:
  friend class kitzoo::core::Singleton<DvppRuntime>;

  DvppRuntime();

  bool ready_ = false;
};

} // namespace hrt::backend

#endif // HRT_ASCEND_ACL_COMMON_HPP

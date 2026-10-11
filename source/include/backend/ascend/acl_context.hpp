// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/ascend/acl_context.hpp
// Description: Declares the Ascend device context and stream.
// -----------------------------------------------------------------------------

#ifndef HRT_ASCEND_ACL_CONTEXT_HPP
#define HRT_ASCEND_ACL_CONTEXT_HPP

#include <hrt/core/context.hpp>

namespace hrt::backend {

class AclStream final : public core::Stream {
public:
  AclStream();

  ~AclStream() override;

  AclStream(const AclStream&) = delete;
  auto operator=(const AclStream&) -> AclStream& = delete;
  AclStream(AclStream&&) = delete;
  auto operator=(AclStream&&) -> AclStream& = delete;

  auto synchronize() -> void override;
};

class AclContext final : public core::Context {
public:
  explicit AclContext(int device_id);

  ~AclContext() override;

  AclContext(const AclContext&) = delete;
  auto operator=(const AclContext&) -> AclContext& = delete;
  AclContext(AclContext&&) = delete;
  auto operator=(AclContext&&) -> AclContext& = delete;

  auto make_current() -> void override;

  auto create_stream() -> UniquePtr<core::Stream> override;
};

} // namespace hrt::backend

#endif // HRT_ASCEND_ACL_CONTEXT_HPP

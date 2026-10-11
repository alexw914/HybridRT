// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/core/containers.hpp
// Description: Brings kitzoo's container, string and ownership aliases into
//              hrt; mimalloc serves them as the process allocator.
// -----------------------------------------------------------------------------

#ifndef HRT_CORE_CONTAINERS_HPP
#define HRT_CORE_CONTAINERS_HPP

#include <kitzoo/memory/memory.hpp>

namespace hrt {

using kitzoo::memory::Array;
using kitzoo::memory::Map;
using kitzoo::memory::Pair;
using kitzoo::memory::Set;
using kitzoo::memory::SharedPtr;
using kitzoo::memory::String;
using kitzoo::memory::UniquePtr;
using kitzoo::memory::Vector;

} // namespace hrt

#endif // HRT_CORE_CONTAINERS_HPP

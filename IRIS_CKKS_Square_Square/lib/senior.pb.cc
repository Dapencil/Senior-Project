// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: senior.proto

#include "header/senior.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
class InputVectorDefaultTypeInternal
{
public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<InputVector> _instance;
} _InputVector_default_instance_;
class PredictionCKKSDefaultTypeInternal
{
public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<PredictionCKKS> _instance;
} _PredictionCKKS_default_instance_;
class PredictionBFVDefaultTypeInternal
{
public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<PredictionBFV> _instance;
} _PredictionBFV_default_instance_;
static void InitDefaultsscc_info_InputVector_senior_2eproto()
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void *ptr = &::_InputVector_default_instance_;
    new (ptr)::InputVector();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::InputVector::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_InputVector_senior_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_InputVector_senior_2eproto}, {}};

static void InitDefaultsscc_info_PredictionBFV_senior_2eproto()
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void *ptr = &::_PredictionBFV_default_instance_;
    new (ptr)::PredictionBFV();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::PredictionBFV::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_PredictionBFV_senior_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_PredictionBFV_senior_2eproto}, {}};

static void InitDefaultsscc_info_PredictionCKKS_senior_2eproto()
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void *ptr = &::_PredictionCKKS_default_instance_;
    new (ptr)::PredictionCKKS();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::PredictionCKKS::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_PredictionCKKS_senior_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_PredictionCKKS_senior_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_senior_2eproto[3];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const **file_level_enum_descriptors_senior_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const **file_level_service_descriptors_senior_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_senior_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
    ~0u, // no _has_bits_
    PROTOBUF_FIELD_OFFSET(::InputVector, _internal_metadata_),
    ~0u, // no _extensions_
    ~0u, // no _oneof_case_
    ~0u, // no _weak_field_map_
    PROTOBUF_FIELD_OFFSET(::InputVector, values_),
    ~0u, // no _has_bits_
    PROTOBUF_FIELD_OFFSET(::PredictionCKKS, _internal_metadata_),
    ~0u, // no _extensions_
    ~0u, // no _oneof_case_
    ~0u, // no _weak_field_map_
    PROTOBUF_FIELD_OFFSET(::PredictionCKKS, values_),
    ~0u, // no _has_bits_
    PROTOBUF_FIELD_OFFSET(::PredictionBFV, _internal_metadata_),
    ~0u, // no _extensions_
    ~0u, // no _oneof_case_
    ~0u, // no _weak_field_map_
    PROTOBUF_FIELD_OFFSET(::PredictionBFV, values_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
    {0, -1, sizeof(::InputVector)},
    {6, -1, sizeof(::PredictionCKKS)},
    {12, -1, sizeof(::PredictionBFV)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const *const file_default_instances[] = {
    reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message *>(&::_InputVector_default_instance_),
    reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message *>(&::_PredictionCKKS_default_instance_),
    reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message *>(&::_PredictionBFV_default_instance_),
};

const char descriptor_table_protodef_senior_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
    "\n\014senior.proto\"\035\n\013InputVector\022\016\n\006values\030"
    "\001 \003(\001\" \n\016PredictionCKKS\022\016\n\006values\030\001 \003(\001\""
    "\037\n\rPredictionBFV\022\016\n\006values\030\001 \003(\005b\006proto3";
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable *const descriptor_table_senior_2eproto_deps[1] = {};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase *const descriptor_table_senior_2eproto_sccs[3] = {
    &scc_info_InputVector_senior_2eproto.base,
    &scc_info_PredictionBFV_senior_2eproto.base,
    &scc_info_PredictionCKKS_senior_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_senior_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_senior_2eproto = {
    false,
    false,
    descriptor_table_protodef_senior_2eproto,
    "senior.proto",
    120,
    &descriptor_table_senior_2eproto_once,
    descriptor_table_senior_2eproto_sccs,
    descriptor_table_senior_2eproto_deps,
    3,
    0,
    schemas,
    file_default_instances,
    TableStruct_senior_2eproto::offsets,
    file_level_metadata_senior_2eproto,
    3,
    file_level_enum_descriptors_senior_2eproto,
    file_level_service_descriptors_senior_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_senior_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_senior_2eproto)), true);

// ===================================================================

void InputVector::InitAsDefaultInstance()
{
}
class InputVector::_Internal
{
public:
};

InputVector::InputVector(::PROTOBUF_NAMESPACE_ID::Arena *arena)
    : ::PROTOBUF_NAMESPACE_ID::Message(arena),
      values_(arena)
{
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:InputVector)
}
InputVector::InputVector(const InputVector &from)
    : ::PROTOBUF_NAMESPACE_ID::Message(),
      values_(from.values_)
{
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:InputVector)
}

void InputVector::SharedCtor()
{
}

InputVector::~InputVector()
{
  // @@protoc_insertion_point(destructor:InputVector)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void InputVector::SharedDtor()
{
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void InputVector::ArenaDtor(void *object)
{
  InputVector *_this = reinterpret_cast<InputVector *>(object);
  (void)_this;
}
void InputVector::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena *)
{
}
void InputVector::SetCachedSize(int size) const
{
  _cached_size_.Set(size);
}
const InputVector &InputVector::default_instance()
{
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_InputVector_senior_2eproto.base);
  return *internal_default_instance();
}

void InputVector::Clear()
{
  // @@protoc_insertion_point(message_clear_start:InputVector)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void)cached_has_bits;

  values_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char *InputVector::_InternalParse(const char *ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext *ctx)
{
#define CHK_(x)                     \
  if (PROTOBUF_PREDICT_FALSE(!(x))) \
  goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena *arena = GetArena();
  (void)arena;
  while (!ctx->Done(&ptr))
  {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3)
    {
    // repeated double values = 1;
    case 1:
      if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10))
      {
        ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedDoubleParser(_internal_mutable_values(), ptr, ctx);
        CHK_(ptr);
      }
      else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 9)
      {
        _internal_add_values(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr));
        ptr += sizeof(double);
      }
      else
        goto handle_unusual;
      continue;
    default:
    {
    handle_unusual:
      if ((tag & 7) == 4 || tag == 0)
      {
        ctx->SetLastTag(tag);
        goto success;
      }
      ptr = UnknownFieldParse(tag,
                              _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
                              ptr, ctx);
      CHK_(ptr != nullptr);
      continue;
    }
    } // switch
  }   // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8 *InputVector::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8 *target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream *stream) const
{
  // @@protoc_insertion_point(serialize_to_array_start:InputVector)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void)cached_has_bits;

  // repeated double values = 1;
  if (this->_internal_values_size() > 0)
  {
    target = stream->WriteFixedPacked(1, _internal_values(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields()))
  {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:InputVector)
  return target;
}

size_t InputVector::ByteSizeLong() const
{
  // @@protoc_insertion_point(message_byte_size_start:InputVector)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void)cached_has_bits;

  // repeated double values = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->_internal_values_size());
    size_t data_size = 8UL * count;
    if (data_size > 0)
    {
      total_size += 1 +
                    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
                        static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _values_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields()))
  {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void InputVector::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message &from)
{
  // @@protoc_insertion_point(generalized_merge_from_start:InputVector)
  GOOGLE_DCHECK_NE(&from, this);
  const InputVector *source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<InputVector>(
          &from);
  if (source == nullptr)
  {
    // @@protoc_insertion_point(generalized_merge_from_cast_fail:InputVector)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  }
  else
  {
    // @@protoc_insertion_point(generalized_merge_from_cast_success:InputVector)
    MergeFrom(*source);
  }
}

void InputVector::MergeFrom(const InputVector &from)
{
  // @@protoc_insertion_point(class_specific_merge_from_start:InputVector)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void)cached_has_bits;

  values_.MergeFrom(from.values_);
}

void InputVector::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message &from)
{
  // @@protoc_insertion_point(generalized_copy_from_start:InputVector)
  if (&from == this)
    return;
  Clear();
  MergeFrom(from);
}

void InputVector::CopyFrom(const InputVector &from)
{
  // @@protoc_insertion_point(class_specific_copy_from_start:InputVector)
  if (&from == this)
    return;
  Clear();
  MergeFrom(from);
}

bool InputVector::IsInitialized() const
{
  return true;
}

void InputVector::InternalSwap(InputVector *other)
{
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  values_.InternalSwap(&other->values_);
}

::PROTOBUF_NAMESPACE_ID::Metadata InputVector::GetMetadata() const
{
  return GetMetadataStatic();
}

// ===================================================================

void PredictionCKKS::InitAsDefaultInstance()
{
}
class PredictionCKKS::_Internal
{
public:
};

PredictionCKKS::PredictionCKKS(::PROTOBUF_NAMESPACE_ID::Arena *arena)
    : ::PROTOBUF_NAMESPACE_ID::Message(arena),
      values_(arena)
{
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:PredictionCKKS)
}
PredictionCKKS::PredictionCKKS(const PredictionCKKS &from)
    : ::PROTOBUF_NAMESPACE_ID::Message(),
      values_(from.values_)
{
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:PredictionCKKS)
}

void PredictionCKKS::SharedCtor()
{
}

PredictionCKKS::~PredictionCKKS()
{
  // @@protoc_insertion_point(destructor:PredictionCKKS)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void PredictionCKKS::SharedDtor()
{
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void PredictionCKKS::ArenaDtor(void *object)
{
  PredictionCKKS *_this = reinterpret_cast<PredictionCKKS *>(object);
  (void)_this;
}
void PredictionCKKS::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena *)
{
}
void PredictionCKKS::SetCachedSize(int size) const
{
  _cached_size_.Set(size);
}
const PredictionCKKS &PredictionCKKS::default_instance()
{
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_PredictionCKKS_senior_2eproto.base);
  return *internal_default_instance();
}

void PredictionCKKS::Clear()
{
  // @@protoc_insertion_point(message_clear_start:PredictionCKKS)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void)cached_has_bits;

  values_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char *PredictionCKKS::_InternalParse(const char *ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext *ctx)
{
#define CHK_(x)                     \
  if (PROTOBUF_PREDICT_FALSE(!(x))) \
  goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena *arena = GetArena();
  (void)arena;
  while (!ctx->Done(&ptr))
  {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3)
    {
    // repeated double values = 1;
    case 1:
      if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10))
      {
        ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedDoubleParser(_internal_mutable_values(), ptr, ctx);
        CHK_(ptr);
      }
      else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 9)
      {
        _internal_add_values(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr));
        ptr += sizeof(double);
      }
      else
        goto handle_unusual;
      continue;
    default:
    {
    handle_unusual:
      if ((tag & 7) == 4 || tag == 0)
      {
        ctx->SetLastTag(tag);
        goto success;
      }
      ptr = UnknownFieldParse(tag,
                              _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
                              ptr, ctx);
      CHK_(ptr != nullptr);
      continue;
    }
    } // switch
  }   // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8 *PredictionCKKS::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8 *target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream *stream) const
{
  // @@protoc_insertion_point(serialize_to_array_start:PredictionCKKS)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void)cached_has_bits;

  // repeated double values = 1;
  if (this->_internal_values_size() > 0)
  {
    target = stream->WriteFixedPacked(1, _internal_values(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields()))
  {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:PredictionCKKS)
  return target;
}

size_t PredictionCKKS::ByteSizeLong() const
{
  // @@protoc_insertion_point(message_byte_size_start:PredictionCKKS)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void)cached_has_bits;

  // repeated double values = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->_internal_values_size());
    size_t data_size = 8UL * count;
    if (data_size > 0)
    {
      total_size += 1 +
                    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
                        static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _values_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields()))
  {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void PredictionCKKS::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message &from)
{
  // @@protoc_insertion_point(generalized_merge_from_start:PredictionCKKS)
  GOOGLE_DCHECK_NE(&from, this);
  const PredictionCKKS *source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<PredictionCKKS>(
          &from);
  if (source == nullptr)
  {
    // @@protoc_insertion_point(generalized_merge_from_cast_fail:PredictionCKKS)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  }
  else
  {
    // @@protoc_insertion_point(generalized_merge_from_cast_success:PredictionCKKS)
    MergeFrom(*source);
  }
}

void PredictionCKKS::MergeFrom(const PredictionCKKS &from)
{
  // @@protoc_insertion_point(class_specific_merge_from_start:PredictionCKKS)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void)cached_has_bits;

  values_.MergeFrom(from.values_);
}

void PredictionCKKS::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message &from)
{
  // @@protoc_insertion_point(generalized_copy_from_start:PredictionCKKS)
  if (&from == this)
    return;
  Clear();
  MergeFrom(from);
}

void PredictionCKKS::CopyFrom(const PredictionCKKS &from)
{
  // @@protoc_insertion_point(class_specific_copy_from_start:PredictionCKKS)
  if (&from == this)
    return;
  Clear();
  MergeFrom(from);
}

bool PredictionCKKS::IsInitialized() const
{
  return true;
}

void PredictionCKKS::InternalSwap(PredictionCKKS *other)
{
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  values_.InternalSwap(&other->values_);
}

::PROTOBUF_NAMESPACE_ID::Metadata PredictionCKKS::GetMetadata() const
{
  return GetMetadataStatic();
}

// ===================================================================

void PredictionBFV::InitAsDefaultInstance()
{
}
class PredictionBFV::_Internal
{
public:
};

PredictionBFV::PredictionBFV(::PROTOBUF_NAMESPACE_ID::Arena *arena)
    : ::PROTOBUF_NAMESPACE_ID::Message(arena),
      values_(arena)
{
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:PredictionBFV)
}
PredictionBFV::PredictionBFV(const PredictionBFV &from)
    : ::PROTOBUF_NAMESPACE_ID::Message(),
      values_(from.values_)
{
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:PredictionBFV)
}

void PredictionBFV::SharedCtor()
{
}

PredictionBFV::~PredictionBFV()
{
  // @@protoc_insertion_point(destructor:PredictionBFV)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void PredictionBFV::SharedDtor()
{
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void PredictionBFV::ArenaDtor(void *object)
{
  PredictionBFV *_this = reinterpret_cast<PredictionBFV *>(object);
  (void)_this;
}
void PredictionBFV::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena *)
{
}
void PredictionBFV::SetCachedSize(int size) const
{
  _cached_size_.Set(size);
}
const PredictionBFV &PredictionBFV::default_instance()
{
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_PredictionBFV_senior_2eproto.base);
  return *internal_default_instance();
}

void PredictionBFV::Clear()
{
  // @@protoc_insertion_point(message_clear_start:PredictionBFV)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void)cached_has_bits;

  values_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char *PredictionBFV::_InternalParse(const char *ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext *ctx)
{
#define CHK_(x)                     \
  if (PROTOBUF_PREDICT_FALSE(!(x))) \
  goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena *arena = GetArena();
  (void)arena;
  while (!ctx->Done(&ptr))
  {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3)
    {
    // repeated int32 values = 1;
    case 1:
      if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10))
      {
        ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_values(), ptr, ctx);
        CHK_(ptr);
      }
      else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)
      {
        _internal_add_values(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
        CHK_(ptr);
      }
      else
        goto handle_unusual;
      continue;
    default:
    {
    handle_unusual:
      if ((tag & 7) == 4 || tag == 0)
      {
        ctx->SetLastTag(tag);
        goto success;
      }
      ptr = UnknownFieldParse(tag,
                              _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
                              ptr, ctx);
      CHK_(ptr != nullptr);
      continue;
    }
    } // switch
  }   // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8 *PredictionBFV::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8 *target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream *stream) const
{
  // @@protoc_insertion_point(serialize_to_array_start:PredictionBFV)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void)cached_has_bits;

  // repeated int32 values = 1;
  {
    int byte_size = _values_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0)
    {
      target = stream->WriteInt32Packed(
          1, _internal_values(), byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields()))
  {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:PredictionBFV)
  return target;
}

size_t PredictionBFV::ByteSizeLong() const
{
  // @@protoc_insertion_point(message_byte_size_start:PredictionBFV)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void)cached_has_bits;

  // repeated int32 values = 1;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        Int32Size(this->values_);
    if (data_size > 0)
    {
      total_size += 1 +
                    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
                        static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _values_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields()))
  {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void PredictionBFV::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message &from)
{
  // @@protoc_insertion_point(generalized_merge_from_start:PredictionBFV)
  GOOGLE_DCHECK_NE(&from, this);
  const PredictionBFV *source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<PredictionBFV>(
          &from);
  if (source == nullptr)
  {
    // @@protoc_insertion_point(generalized_merge_from_cast_fail:PredictionBFV)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  }
  else
  {
    // @@protoc_insertion_point(generalized_merge_from_cast_success:PredictionBFV)
    MergeFrom(*source);
  }
}

void PredictionBFV::MergeFrom(const PredictionBFV &from)
{
  // @@protoc_insertion_point(class_specific_merge_from_start:PredictionBFV)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void)cached_has_bits;

  values_.MergeFrom(from.values_);
}

void PredictionBFV::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message &from)
{
  // @@protoc_insertion_point(generalized_copy_from_start:PredictionBFV)
  if (&from == this)
    return;
  Clear();
  MergeFrom(from);
}

void PredictionBFV::CopyFrom(const PredictionBFV &from)
{
  // @@protoc_insertion_point(class_specific_copy_from_start:PredictionBFV)
  if (&from == this)
    return;
  Clear();
  MergeFrom(from);
}

bool PredictionBFV::IsInitialized() const
{
  return true;
}

void PredictionBFV::InternalSwap(PredictionBFV *other)
{
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  values_.InternalSwap(&other->values_);
}

::PROTOBUF_NAMESPACE_ID::Metadata PredictionBFV::GetMetadata() const
{
  return GetMetadataStatic();
}

// @@protoc_insertion_point(namespace_scope)
PROTOBUF_NAMESPACE_OPEN
template <>
PROTOBUF_NOINLINE ::InputVector *Arena::CreateMaybeMessage<::InputVector>(Arena *arena)
{
  return Arena::CreateMessageInternal<::InputVector>(arena);
}
template <>
PROTOBUF_NOINLINE ::PredictionCKKS *Arena::CreateMaybeMessage<::PredictionCKKS>(Arena *arena)
{
  return Arena::CreateMessageInternal<::PredictionCKKS>(arena);
}
template <>
PROTOBUF_NOINLINE ::PredictionBFV *Arena::CreateMaybeMessage<::PredictionBFV>(Arena *arena)
{
  return Arena::CreateMessageInternal<::PredictionBFV>(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
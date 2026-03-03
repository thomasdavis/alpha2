# VK_EXT_device_generated_commands -- Complete Specification Reference

> Combined from the Vulkan specification (registry.khronos.org), the Khronos proposal
> document (Vulkan-Docs/proposals/VK_EXT_device_generated_commands.adoc), the LunarG
> SDK reference, and the official Vulkan spec chapters. Last updated 2026-03-03.

---

## Table of Contents

1. [Extension Metadata](#1-extension-metadata)
2. [Overview](#2-overview)
3. [Key Differences from VK_NV_device_generated_commands](#3-key-differences-from-vk_nv_device_generated_commands)
4. [Feature and Property Queries](#4-feature-and-property-queries)
5. [Indirect Execution Sets](#5-indirect-execution-sets)
6. [Indirect Commands Layout](#6-indirect-commands-layout)
7. [Token Types and Data Structures](#7-token-types-and-data-structures)
8. [Preprocess Buffer and Memory Requirements](#8-preprocess-buffer-and-memory-requirements)
9. [Command Execution](#9-command-execution)
10. [Synchronization](#10-synchronization)
11. [Valid Usage Rules](#11-valid-usage-rules)
12. [D3D12 Emulation Mappings](#12-d3d12-emulation-mappings)
13. [VkStructureType Enum Values](#13-vkstructuretype-enum-values)
14. [Pipeline and Shader Creation Flags](#14-pipeline-and-shader-creation-flags)
15. [Usage Workflow](#15-usage-workflow)
16. [Unresolved Design Questions](#16-unresolved-design-questions)

---

## 1. Extension Metadata

| Field | Value |
|---|---|
| **Extension Name** | `VK_EXT_device_generated_commands` |
| **Extension Type** | Device extension |
| **Registered Extension Number** | 573 |
| **Revision** | 1 |
| **Dependencies** | Vulkan 1.1, `VK_KHR_buffer_device_address` or Vulkan 1.2 |
| **Optional Dependencies** | `VK_EXT_shader_object`, `VK_EXT_mesh_shader`, `VK_KHR_ray_tracing_maintenance1` |
| **Contact** | Christoph Kubisch (NVIDIA) |

---

## 2. Overview

`VK_EXT_device_generated_commands` enables device-side generation and execution of command
sequences, reducing unnecessary state changes and improving performance in GPU-driven
rendering scenarios. It allows compute shaders to fill a buffer with command tokens that
the implementation reads at execution time, rather than requiring the host to record
every command into a command buffer.

The extension provides a unified framework for indirect execution supporting graphics,
compute, and ray tracing commands through three core components:

1. **`VkIndirectExecutionSetEXT`** -- A collection managing shaders/pipelines for
   indirect binding. Similar to descriptor sets, it serves as a binding table with a
   fixed upper count.
2. **`VkIndirectCommandsLayoutEXT`** -- Describes the command sequence structure in
   buffers (a fixed-size, homogeneous layout).
3. **Preprocess and Execute pipeline** -- Separate logical pipelines for optimization:
   preprocessing can be explicit (on a separate queue) or implicit (auto-synchronized).

### Conceptual Model

Rather than the host recording individual draw/dispatch calls:

```
// Traditional: Host records N commands
vkCmdPushConstants(...)
vkCmdDispatch(x, y, z)
vkCmdPushConstants(...)
vkCmdDispatch(x, y, z)
...
```

With DGC, the GPU fills a buffer matching a declared layout, and the implementation
reads command sequences from it:

```
// DGC: GPU fills buffer, implementation reads sequences
VkIndirectCommandsLayoutEXT layout:
  1) PUSH_CONSTANT token
  2) DISPATCH token

Buffer contains N fixed-size sequences:
  [push_constant_data_0 | dispatch_xyz_0]
  [push_constant_data_1 | dispatch_xyz_1]
  ...
```

Each sequence is `indirectStride` bytes. Data is read at:
```
offset = stride * sequenceIndex + tokenOffset
```

---

## 3. Key Differences from VK_NV_device_generated_commands

| Feature | NV | EXT |
|---|---|---|
| Unified framework (graphics + compute + ray tracing) | No | **Yes** |
| Incremental shader updates via execution sets | No | **Yes** |
| IndirectCount commands | No | **Yes** |
| Compute dispatch support | Via separate ext | **Built-in** |
| Single-interleaved stream | Multi-stream | **Single stream** |
| `VK_EXT_shader_object` support | No | **Yes** |
| Shader group binding | Yes (`SHADER_GROUP_NV`) | Via `EXECUTION_SET_EXT` |
| State flags (front face toggle) | Yes (`STATE_FLAGS_NV`) | No |
| Index type remapping | Yes (`pIndexTypeValues`) | No (use `mode` flag) |

---

## 4. Feature and Property Queries

### 4.1 VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT

Chain into `VkPhysicalDeviceFeatures2` / `VkDeviceCreateInfo`.

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT {
    VkStructureType    sType;       // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT
    void*              pNext;
    VkBool32           deviceGeneratedCommands;
    VkBool32           dynamicGeneratedPipelineLayout;
} VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT;
```

| Member | Description |
|---|---|
| `deviceGeneratedCommands` | Core feature gate. Must be enabled to use any functionality in this extension. |
| `dynamicGeneratedPipelineLayout` | If `VK_TRUE`, `pipelineLayout` in `VkIndirectCommandsLayoutCreateInfoEXT` can be `VK_NULL_HANDLE` for push constant and sequence index tokens. The layout is inferred dynamically. |

### 4.2 VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT

Chain into `VkPhysicalDeviceProperties2`.

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT {
    VkStructureType                      sType;   // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT
    void*                                pNext;
    uint32_t                             maxIndirectPipelineCount;
    uint32_t                             maxIndirectShaderObjectCount;
    uint32_t                             maxIndirectSequenceCount;
    uint32_t                             maxIndirectCommandsTokenCount;
    uint32_t                             maxIndirectCommandsTokenOffset;
    uint32_t                             maxIndirectCommandsIndirectStride;
    VkIndirectCommandsInputModeFlagsEXT  supportedIndirectCommandsInputModes;
    VkShaderStageFlags                   supportedIndirectCommandsShaderStages;
    VkShaderStageFlags                   supportedIndirectCommandsShaderStagesPipelineBinding;
    VkShaderStageFlags                   supportedIndirectCommandsShaderStagesShaderBinding;
    VkBool32                             deviceGeneratedCommandsTransformFeedback;
    VkBool32                             deviceGeneratedCommandsMultiDrawIndirectCount;
} VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT;
```

| Member | Description |
|---|---|
| `maxIndirectPipelineCount` | Maximum number of pipelines in a pipeline-type `VkIndirectExecutionSetEXT`. |
| `maxIndirectShaderObjectCount` | Maximum number of shader objects in a shader-object-type `VkIndirectExecutionSetEXT`. |
| `maxIndirectSequenceCount` | Maximum value of `maxSequenceCount` in `VkGeneratedCommandsInfoEXT`. |
| `maxIndirectCommandsTokenCount` | Maximum number of tokens per indirect commands layout. |
| `maxIndirectCommandsTokenOffset` | Maximum byte offset for any token in a layout. |
| `maxIndirectCommandsIndirectStride` | Maximum stride between sequences in the indirect buffer. |
| `supportedIndirectCommandsInputModes` | Bitmask of `VkIndirectCommandsInputModeFlagBitsEXT` supported for index buffer tokens. |
| `supportedIndirectCommandsShaderStages` | All shader stages that can appear in any indirect commands layout. |
| `supportedIndirectCommandsShaderStagesPipelineBinding` | Shader stages supported when using pipeline-based execution sets. |
| `supportedIndirectCommandsShaderStagesShaderBinding` | Shader stages supported when using shader-object-based execution sets. |
| `deviceGeneratedCommandsTransformFeedback` | Whether transform feedback is supported during DGC execution. |
| `deviceGeneratedCommandsMultiDrawIndirectCount` | Whether `*_COUNT_*` token types are supported. |

---

## 5. Indirect Execution Sets

An `VkIndirectExecutionSetEXT` is a binding table of pipelines or shader objects. The
DGC buffer contains indices into this table, allowing the GPU to select which
pipeline/shader to use per sequence.

### 5.1 Handle

```c
// Provided by VK_EXT_device_generated_commands
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkIndirectExecutionSetEXT)
```

### 5.2 Creation

```c
// Provided by VK_EXT_device_generated_commands
VKAPI_ATTR VkResult VKAPI_CALL vkCreateIndirectExecutionSetEXT(
    VkDevice                                    device,
    const VkIndirectExecutionSetCreateInfoEXT*   pCreateInfo,
    const VkAllocationCallbacks*                 pAllocator,
    VkIndirectExecutionSetEXT*                   pIndirectExecutionSet);
```

**Return Codes:**
- `VK_SUCCESS`
- `VK_ERROR_OUT_OF_HOST_MEMORY`
- `VK_ERROR_OUT_OF_DEVICE_MEMORY`

### 5.3 VkIndirectExecutionSetCreateInfoEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkIndirectExecutionSetCreateInfoEXT {
    VkStructureType                       sType;   // VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT
    const void*                           pNext;
    VkIndirectExecutionSetInfoTypeEXT     type;
    VkIndirectExecutionSetInfoEXT         info;
} VkIndirectExecutionSetCreateInfoEXT;
```

| Member | Description |
|---|---|
| `type` | Whether this set holds pipelines or shader objects. |
| `info` | Union selecting the appropriate info struct based on `type`. |

### 5.4 VkIndirectExecutionSetInfoTypeEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef enum VkIndirectExecutionSetInfoTypeEXT {
    VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT       = 0,
    VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT  = 1,
} VkIndirectExecutionSetInfoTypeEXT;
```

### 5.5 VkIndirectExecutionSetInfoEXT (Union)

```c
// Provided by VK_EXT_device_generated_commands
typedef union VkIndirectExecutionSetInfoEXT {
    const VkIndirectExecutionSetPipelineInfoEXT*  pPipelineInfo;
    const VkIndirectExecutionSetShaderInfoEXT*    pShaderInfo;
} VkIndirectExecutionSetInfoEXT;
```

- If `type == VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT`, use `pPipelineInfo`.
- If `type == VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT`, use `pShaderInfo`.

### 5.6 VkIndirectExecutionSetPipelineInfoEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkIndirectExecutionSetPipelineInfoEXT {
    VkStructureType    sType;   // VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT
    const void*        pNext;
    VkPipeline         initialPipeline;
    uint32_t           maxPipelineCount;
} VkIndirectExecutionSetPipelineInfoEXT;
```

| Member | Description |
|---|---|
| `initialPipeline` | Pipeline stored at index 0. Must have been created with `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`. Defines the layout compatibility requirements for all pipelines in the set. |
| `maxPipelineCount` | Maximum number of pipeline slots. Must be <= `maxIndirectPipelineCount`. |

### 5.7 VkIndirectExecutionSetShaderInfoEXT

```c
// Provided by VK_EXT_device_generated_commands with VK_EXT_shader_object
typedef struct VkIndirectExecutionSetShaderInfoEXT {
    VkStructureType                                      sType;   // VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_INFO_EXT
    const void*                                          pNext;
    uint32_t                                             shaderCount;
    const VkShaderEXT*                                   pInitialShaders;
    const VkIndirectExecutionSetShaderLayoutInfoEXT*     pSetLayoutInfos;
    uint32_t                                             maxShaderCount;
    uint32_t                                             pushConstantRangeCount;
    const VkPushConstantRange*                           pPushConstantRanges;
} VkIndirectExecutionSetShaderInfoEXT;
```

| Member | Description |
|---|---|
| `shaderCount` | Number of initial shaders. |
| `pInitialShaders` | Array of `shaderCount` shader objects to populate the first slots. Must have been created with `VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT`. |
| `pSetLayoutInfos` | Per-shader descriptor set layout info. |
| `maxShaderCount` | Maximum number of shader slots. Must be <= `maxIndirectShaderObjectCount`. |
| `pushConstantRangeCount` | Number of push constant ranges used by any shader in the set. |
| `pPushConstantRanges` | Array of push constant ranges (superset of all shaders). |

### 5.8 VkIndirectExecutionSetShaderLayoutInfoEXT

```c
// Provided by VK_EXT_device_generated_commands with VK_EXT_shader_object
typedef struct VkIndirectExecutionSetShaderLayoutInfoEXT {
    VkStructureType              sType;   // VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_LAYOUT_INFO_EXT
    const void*                  pNext;
    uint32_t                     setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
} VkIndirectExecutionSetShaderLayoutInfoEXT;
```

### 5.9 Destruction

```c
// Provided by VK_EXT_device_generated_commands
VKAPI_ATTR void VKAPI_CALL vkDestroyIndirectExecutionSetEXT(
    VkDevice                        device,
    VkIndirectExecutionSetEXT       indirectExecutionSet,
    const VkAllocationCallbacks*    pAllocator);
```

### 5.10 Updating Pipelines in a Set

```c
// Provided by VK_EXT_device_generated_commands
VKAPI_ATTR void VKAPI_CALL vkUpdateIndirectExecutionSetPipelineEXT(
    VkDevice                                        device,
    VkIndirectExecutionSetEXT                       indirectExecutionSet,
    uint32_t                                        executionSetWriteCount,
    const VkWriteIndirectExecutionSetPipelineEXT*   pExecutionSetWrites);
```

### 5.11 VkWriteIndirectExecutionSetPipelineEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkWriteIndirectExecutionSetPipelineEXT {
    VkStructureType    sType;   // VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT
    const void*        pNext;
    uint32_t           index;
    VkPipeline         pipeline;
} VkWriteIndirectExecutionSetPipelineEXT;
```

| Member | Description |
|---|---|
| `index` | Slot index in the execution set to update. |
| `pipeline` | Pipeline to store. Must have been created with `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT` and be layout-compatible with `initialPipeline`. |

### 5.12 Updating Shaders in a Set

```c
// Provided by VK_EXT_device_generated_commands with VK_EXT_shader_object
VKAPI_ATTR void VKAPI_CALL vkUpdateIndirectExecutionSetShaderEXT(
    VkDevice                                       device,
    VkIndirectExecutionSetEXT                      indirectExecutionSet,
    uint32_t                                       executionSetWriteCount,
    const VkWriteIndirectExecutionSetShaderEXT*    pExecutionSetWrites);
```

### 5.13 VkWriteIndirectExecutionSetShaderEXT

```c
// Provided by VK_EXT_device_generated_commands with VK_EXT_shader_object
typedef struct VkWriteIndirectExecutionSetShaderEXT {
    VkStructureType    sType;   // VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_SHADER_EXT
    const void*        pNext;
    uint32_t           index;
    VkShaderEXT        shader;
} VkWriteIndirectExecutionSetShaderEXT;
```

### 5.14 Execution Set Update Rules

- Slots can be modified when the execution set is **not in-flight** (no submitted
  command buffer references it).
- Changes may alter preprocessing memory requirements. After modifying slots, call
  `vkGetGeneratedCommandsMemoryRequirementsEXT` and resize the preprocess buffer
  if needed.
- Drivers should ensure updating a set is a **cheap operation**.

### 5.15 Execution Set Binding Table

| Type | Stored Objects | Update Function | Pipeline/Shader Flag |
|---|---|---|---|
| `VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT` | `VkPipeline` | `vkUpdateIndirectExecutionSetPipelineEXT` | `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT` |
| `VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT` | `VkShaderEXT` | `vkUpdateIndirectExecutionSetShaderEXT` | `VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT` |

The `VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT` token requires either a single
`uint32_t` (pipeline index) or N `uint32_t` values (one per shader stage) in the
indirect buffer to select which pipeline/shaders to bind for the current sequence.

---

## 6. Indirect Commands Layout

### 6.1 Handle

```c
// Provided by VK_EXT_device_generated_commands
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkIndirectCommandsLayoutEXT)
```

### 6.2 Creation

```c
// Provided by VK_EXT_device_generated_commands
VKAPI_ATTR VkResult VKAPI_CALL vkCreateIndirectCommandsLayoutEXT(
    VkDevice                                        device,
    const VkIndirectCommandsLayoutCreateInfoEXT*    pCreateInfo,
    const VkAllocationCallbacks*                    pAllocator,
    VkIndirectCommandsLayoutEXT*                    pIndirectCommandsLayout);
```

**Requirement:** `deviceGeneratedCommands` feature must be enabled.

**Return Codes:**
- `VK_SUCCESS`
- `VK_ERROR_OUT_OF_HOST_MEMORY`
- `VK_ERROR_OUT_OF_DEVICE_MEMORY`

### 6.3 VkIndirectCommandsLayoutCreateInfoEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkIndirectCommandsLayoutCreateInfoEXT {
    VkStructureType                           sType;   // VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT
    const void*                               pNext;
    VkIndirectCommandsLayoutUsageFlagsEXT     flags;
    VkShaderStageFlags                        shaderStages;
    uint32_t                                  indirectStride;
    VkPipelineLayout                          pipelineLayout;
    uint32_t                                  tokenCount;
    const VkIndirectCommandsLayoutTokenEXT*   pTokens;
} VkIndirectCommandsLayoutCreateInfoEXT;
```

| Member | Description |
|---|---|
| `flags` | Bitmask of `VkIndirectCommandsLayoutUsageFlagBitsEXT`. |
| `shaderStages` | Shader stages this layout targets. Must be a subset of `supportedIndirectCommandsShaderStages`. |
| `indirectStride` | Byte stride between consecutive sequences in the indirect buffer. Must be <= `maxIndirectCommandsIndirectStride`. |
| `pipelineLayout` | Pipeline layout for push constant tokens. May be `VK_NULL_HANDLE` if `dynamicGeneratedPipelineLayout` is enabled and no push constant tokens are used. |
| `tokenCount` | Number of tokens. Must be <= `maxIndirectCommandsTokenCount`. |
| `pTokens` | Array of `tokenCount` token definitions. |

### 6.4 VkIndirectCommandsLayoutUsageFlagBitsEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef enum VkIndirectCommandsLayoutUsageFlagBitsEXT {
    VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT   = 0x00000001,
    VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_EXT   = 0x00000002,
} VkIndirectCommandsLayoutUsageFlagBitsEXT;
typedef VkFlags VkIndirectCommandsLayoutUsageFlagsEXT;
```

| Flag | Description |
|---|---|
| `EXPLICIT_PREPROCESS_BIT_EXT` | This layout always uses manual preprocessing via `vkCmdPreprocessGeneratedCommandsEXT`. The implementation will not auto-preprocess at execute time. |
| `UNORDERED_SEQUENCES_BIT_EXT` | Sequences may be processed in implementation-dependent order. Ignored for compute dispatches. Enables potential optimization. |

### 6.5 Destruction

```c
// Provided by VK_EXT_device_generated_commands
VKAPI_ATTR void VKAPI_CALL vkDestroyIndirectCommandsLayoutEXT(
    VkDevice                        device,
    VkIndirectCommandsLayoutEXT     indirectCommandsLayout,
    const VkAllocationCallbacks*    pAllocator);
```

---

## 7. Token Types and Data Structures

### 7.1 VkIndirectCommandsLayoutTokenEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkIndirectCommandsLayoutTokenEXT {
    VkStructureType                    sType;   // VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT
    const void*                        pNext;
    VkIndirectCommandsTokenTypeEXT     type;
    VkIndirectCommandsTokenDataEXT     data;
    uint32_t                           offset;
} VkIndirectCommandsLayoutTokenEXT;
```

| Member | Description |
|---|---|
| `type` | The token command type (enum value from `VkIndirectCommandsTokenTypeEXT`). |
| `data` | Union providing additional type-specific configuration. |
| `offset` | Byte offset of this token's data within each sequence. Must be 4-byte aligned and <= `maxIndirectCommandsTokenOffset`. |

### 7.2 VkIndirectCommandsTokenTypeEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef enum VkIndirectCommandsTokenTypeEXT {
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT           = 0,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT           = 1,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_SEQUENCE_INDEX_EXT          = 2,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_INDEX_BUFFER_EXT            = 3,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_VERTEX_BUFFER_EXT           = 4,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_EXT            = 5,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_EXT                    = 6,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_COUNT_EXT      = 7,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_COUNT_EXT              = 8,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT                = 9,

    // Provided by VK_EXT_device_generated_commands with VK_NV_mesh_shader
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_NV_EXT          = 1000202002,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_NV_EXT    = 1000202003,

    // Provided by VK_EXT_device_generated_commands with VK_EXT_mesh_shader
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_EXT             = 1000328000,
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_EXT       = 1000328001,

    // Provided by VK_EXT_device_generated_commands with VK_KHR_ray_tracing_maintenance1
    VK_INDIRECT_COMMANDS_TOKEN_TYPE_TRACE_RAYS2_EXT                 = 1000386004,
} VkIndirectCommandsTokenTypeEXT;
```

**Token classification:**

| Token | Category | Indirect Buffer Data Size | Description |
|---|---|---|---|
| `EXECUTION_SET_EXT` | State | `uint32_t` (pipeline index) or N x `uint32_t` (per-stage shader index) | Binds pipeline/shader from execution set. **Must be first token if present.** |
| `PUSH_CONSTANT_EXT` | State | `updateRange.size` bytes | Writes push constant data. |
| `SEQUENCE_INDEX_EXT` | State | (implicit, not in buffer) | Writes `sequenceIndex` as a `uint32_t` push constant at `updateRange.offset`. |
| `INDEX_BUFFER_EXT` | State | `sizeof(VkBindIndexBufferIndirectCommandEXT)` = 16 bytes | Binds an index buffer. |
| `VERTEX_BUFFER_EXT` | State | `sizeof(VkBindVertexBufferIndirectCommandEXT)` = 16 bytes | Binds a vertex buffer. |
| `DRAW_INDEXED_EXT` | Action | `sizeof(VkDrawIndexedIndirectCommand)` = 20 bytes | Indexed draw. |
| `DRAW_EXT` | Action | `sizeof(VkDrawIndirectCommand)` = 16 bytes | Non-indexed draw. |
| `DRAW_INDEXED_COUNT_EXT` | Action | `sizeof(VkDrawIndirectCountIndirectCommandEXT)` = 16 bytes | Multi-draw indexed with device-specified count. |
| `DRAW_COUNT_EXT` | Action | `sizeof(VkDrawIndirectCountIndirectCommandEXT)` = 16 bytes | Multi-draw with device-specified count. |
| `DISPATCH_EXT` | Action | `sizeof(VkDispatchIndirectCommand)` = 12 bytes | Compute dispatch. |
| `DRAW_MESH_TASKS_EXT` | Action | `sizeof(VkDrawMeshTasksIndirectCommandEXT)` | Mesh shader draw (EXT). |
| `DRAW_MESH_TASKS_NV_EXT` | Action | `sizeof(VkDrawMeshTasksIndirectCommandNV)` | Mesh shader draw (NV). |
| `DRAW_MESH_TASKS_COUNT_EXT` | Action | `sizeof(VkDrawIndirectCountIndirectCommandEXT)` | Mesh shader multi-draw with count (EXT). |
| `DRAW_MESH_TASKS_COUNT_NV_EXT` | Action | `sizeof(VkDrawIndirectCountIndirectCommandEXT)` | Mesh shader multi-draw with count (NV). |
| `TRACE_RAYS2_EXT` | Action | `sizeof(VkTraceRaysIndirectCommand2KHR)` | Ray tracing dispatch. |

### 7.3 VkIndirectCommandsTokenDataEXT (Union)

```c
// Provided by VK_EXT_device_generated_commands
typedef union VkIndirectCommandsTokenDataEXT {
    const VkIndirectCommandsPushConstantTokenEXT*    pPushConstant;
    const VkIndirectCommandsVertexBufferTokenEXT*    pVertexBuffer;
    const VkIndirectCommandsIndexBufferTokenEXT*     pIndexBuffer;
    const VkIndirectCommandsExecutionSetTokenEXT*    pExecutionSet;
} VkIndirectCommandsTokenDataEXT;
```

**Usage by token type:**
- `EXECUTION_SET_EXT` -> `pExecutionSet`
- `PUSH_CONSTANT_EXT`, `SEQUENCE_INDEX_EXT` -> `pPushConstant`
- `INDEX_BUFFER_EXT` -> `pIndexBuffer`
- `VERTEX_BUFFER_EXT` -> `pVertexBuffer`
- Action tokens (`DRAW_*`, `DISPATCH_EXT`, `TRACE_RAYS2_EXT`) -> union is unused (no extra config needed)

**Note:** "New pointer members will be added to `VkIndirectCommandsTokenDataEXT`" for
future command types.

### 7.4 VkIndirectCommandsExecutionSetTokenEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkIndirectCommandsExecutionSetTokenEXT {
    VkIndirectExecutionSetInfoTypeEXT    type;
    VkShaderStageFlags                   shaderStages;
} VkIndirectCommandsExecutionSetTokenEXT;
```

| Member | Description |
|---|---|
| `type` | Must be `PIPELINES_EXT` or `SHADER_OBJECTS_EXT`. Must match the execution set type used at execution time. |
| `shaderStages` | Which shader stages are bound. For pipeline binding, typically all stages of the pipeline. For shader object binding, each stage gets a separate index in the buffer. |

**This must be the first command token in a sequence when used.**

### 7.5 VkIndirectCommandsPushConstantTokenEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkIndirectCommandsPushConstantTokenEXT {
    VkPushConstantRange    updateRange;
} VkIndirectCommandsPushConstantTokenEXT;
```

| Member | Description |
|---|---|
| `updateRange` | The push constant range to update. `offset` and `size` must be 4-byte aligned. The range must be within the push constant info of the pipeline layout. |

For `SEQUENCE_INDEX_EXT` token, `updateRange.size` must be exactly 4 bytes (`uint32_t`).

### 7.6 VkIndirectCommandsIndexBufferTokenEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkIndirectCommandsIndexBufferTokenEXT {
    VkIndirectCommandsInputModeFlagsEXT    mode;
} VkIndirectCommandsIndexBufferTokenEXT;
```

| Member | Description |
|---|---|
| `mode` | A single `VkIndirectCommandsInputModeFlagBitsEXT` value. Must be supported in `supportedIndirectCommandsInputModes`. |

### 7.7 VkIndirectCommandsInputModeFlagBitsEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef enum VkIndirectCommandsInputModeFlagBitsEXT {
    VK_INDIRECT_COMMANDS_INPUT_MODE_VULKAN_INDEX_BUFFER_EXT  = 0x00000001,
    VK_INDIRECT_COMMANDS_INPUT_MODE_DXGI_INDEX_BUFFER_EXT    = 0x00000002,
} VkIndirectCommandsInputModeFlagBitsEXT;
typedef VkFlags VkIndirectCommandsInputModeFlagsEXT;
```

| Flag | Description |
|---|---|
| `VULKAN_INDEX_BUFFER_EXT` | Buffer data is `VkBindIndexBufferIndirectCommandEXT`. |
| `DXGI_INDEX_BUFFER_EXT` | Buffer data is binary-compatible with `D3D12_INDEX_BUFFER_VIEW` (for D3D12 emulation layers like Proton/DXVK). |

### 7.8 VkIndirectCommandsVertexBufferTokenEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkIndirectCommandsVertexBufferTokenEXT {
    uint32_t    vertexBindingUnit;
} VkIndirectCommandsVertexBufferTokenEXT;
```

| Member | Description |
|---|---|
| `vertexBindingUnit` | The vertex input binding unit this token writes to. Must be less than the total vertex input bindings in use. |

Multiple `VERTEX_BUFFER_EXT` tokens are allowed but each must have a unique
`vertexBindingUnit`.

### 7.9 Indirect Buffer Data Structures

These structures define what the GPU writes/reads per-sequence in the indirect buffer.

#### VkBindIndexBufferIndirectCommandEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkBindIndexBufferIndirectCommandEXT {
    VkDeviceAddress    bufferAddress;    // Physical address of the index buffer
    uint32_t           size;             // Byte size available from bufferAddress
    VkIndexType        indexType;        // VK_INDEX_TYPE_UINT16 or VK_INDEX_TYPE_UINT32
} VkBindIndexBufferIndirectCommandEXT;
```

**Constraints:**
- The buffer at `bufferAddress` must have `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`.
- `bufferAddress` must be aligned per `indexType` requirements.
- Non-sparse buffers must be completely bound to a single `VkDeviceMemory`.

#### VkBindVertexBufferIndirectCommandEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkBindVertexBufferIndirectCommandEXT {
    VkDeviceAddress    bufferAddress;    // Physical address of the vertex buffer
    uint32_t           size;             // Byte size available from bufferAddress
    uint32_t           stride;           // Vertex stride
} VkBindVertexBufferIndirectCommandEXT;
```

**Constraints:**
- The buffer at `bufferAddress` must have `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`.
- Non-sparse buffers must be completely bound to a single `VkDeviceMemory`.

#### VkDrawIndirectCountIndirectCommandEXT

Used by `*_COUNT_*` action tokens (multi-draw with device-specified count).

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkDrawIndirectCountIndirectCommandEXT {
    VkDeviceAddress    bufferAddress;    // Address of buffer containing draw commands
    uint32_t           stride;           // Stride between draw commands in the buffer
    uint32_t           commandCount;     // Number of draw commands
} VkDrawIndirectCountIndirectCommandEXT;
```

**Constraints:**
- The buffer at `bufferAddress` must have `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT`.
- `deviceGeneratedCommandsMultiDrawIndirectCount` must be supported.
- Non-sparse buffers must be completely bound to a single `VkDeviceMemory`.

#### Standard Vulkan Draw Commands (used directly)

For `DRAW_EXT`, the indirect buffer data is `VkDrawIndirectCommand`:
```c
typedef struct VkDrawIndirectCommand {
    uint32_t    vertexCount;
    uint32_t    instanceCount;
    uint32_t    firstVertex;
    uint32_t    firstInstance;
} VkDrawIndirectCommand;   // 16 bytes
```

For `DRAW_INDEXED_EXT`, the indirect buffer data is `VkDrawIndexedIndirectCommand`:
```c
typedef struct VkDrawIndexedIndirectCommand {
    uint32_t    indexCount;
    uint32_t    instanceCount;
    uint32_t    firstIndex;
    int32_t     vertexOffset;
    uint32_t    firstInstance;
} VkDrawIndexedIndirectCommand;   // 20 bytes
```

For `DISPATCH_EXT`, the indirect buffer data is `VkDispatchIndirectCommand`:
```c
typedef struct VkDispatchIndirectCommand {
    uint32_t    x;
    uint32_t    y;
    uint32_t    z;
} VkDispatchIndirectCommand;   // 12 bytes
```

---

## 8. Preprocess Buffer and Memory Requirements

### 8.1 vkGetGeneratedCommandsMemoryRequirementsEXT

```c
// Provided by VK_EXT_device_generated_commands
VKAPI_ATTR void VKAPI_CALL vkGetGeneratedCommandsMemoryRequirementsEXT(
    VkDevice                                              device,
    const VkGeneratedCommandsMemoryRequirementsInfoEXT*   pInfo,
    VkMemoryRequirements2*                                pMemoryRequirements);
```

The application must allocate a preprocess buffer satisfying the returned memory
requirements. The preprocess buffer contents are **opaque to applications** --
implementations use it as scratch space for command preprocessing.

### 8.2 VkGeneratedCommandsMemoryRequirementsInfoEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkGeneratedCommandsMemoryRequirementsInfoEXT {
    VkStructureType                sType;   // VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT
    const void*                    pNext;
    VkIndirectExecutionSetEXT      indirectExecutionSet;
    VkIndirectCommandsLayoutEXT    indirectCommandsLayout;
    uint32_t                       maxSequenceCount;
    uint32_t                       maxDrawCount;
} VkGeneratedCommandsMemoryRequirementsInfoEXT;
```

| Member | Description |
|---|---|
| `indirectExecutionSet` | The execution set that will be used. |
| `indirectCommandsLayout` | The layout that will be used. |
| `maxSequenceCount` | Maximum number of sequences that will be generated. |
| `maxDrawCount` | Maximum draw count for `*_COUNT_*` tokens (0 if not used). |

The `pNext` chain can include:
- `VkGeneratedCommandsPipelineInfoEXT` -- to specify a specific pipeline for tighter memory estimates.
- `VkGeneratedCommandsShaderInfoEXT` -- to specify specific shaders for tighter memory estimates.

### 8.3 VkGeneratedCommandsPipelineInfoEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkGeneratedCommandsPipelineInfoEXT {
    VkStructureType    sType;   // VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT
    const void*        pNext;
    VkPipeline         pipeline;
} VkGeneratedCommandsPipelineInfoEXT;
```

`pipeline` must be compatible with the ones used in the resulting indirect buffer.

### 8.4 VkGeneratedCommandsShaderInfoEXT

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkGeneratedCommandsShaderInfoEXT {
    VkStructureType       sType;   // VK_STRUCTURE_TYPE_GENERATED_COMMANDS_SHADER_INFO_EXT
    const void*           pNext;
    uint32_t              shaderCount;
    const VkShaderEXT*    pShaders;
} VkGeneratedCommandsShaderInfoEXT;
```

---

## 9. Command Execution

### 9.1 VkGeneratedCommandsInfoEXT

This is the primary structure passed to both preprocessing and execution commands.

```c
// Provided by VK_EXT_device_generated_commands
typedef struct VkGeneratedCommandsInfoEXT {
    VkStructureType                sType;   // VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT
    const void*                    pNext;
    VkShaderStageFlags             shaderStages;
    VkIndirectExecutionSetEXT      indirectExecutionSet;
    VkIndirectCommandsLayoutEXT    indirectCommandsLayout;
    VkDeviceAddress                indirectAddress;
    VkDeviceSize                   indirectAddressSize;
    VkDeviceAddress                preprocessAddress;
    VkDeviceSize                   preprocessSize;
    uint32_t                       maxSequenceCount;
    VkDeviceAddress                sequenceCountAddress;
    uint32_t                       maxDrawCount;
} VkGeneratedCommandsInfoEXT;
```

| Member | Description |
|---|---|
| `shaderStages` | Shader stages targeted. Must match the layout's `shaderStages`. |
| `indirectExecutionSet` | Execution set to use for `EXECUTION_SET_EXT` tokens. |
| `indirectCommandsLayout` | Layout describing the sequence structure. |
| `indirectAddress` | Device address of the buffer containing sequence data. |
| `indirectAddressSize` | Size of the indirect data region in bytes. |
| `preprocessAddress` | Device address of the preprocess buffer (opaque scratch space). |
| `preprocessSize` | Size of the preprocess buffer. Must be >= the size returned by `vkGetGeneratedCommandsMemoryRequirementsEXT`. |
| `maxSequenceCount` | Maximum number of sequences to process. Must be <= `maxIndirectSequenceCount`. |
| `sequenceCountAddress` | Device address of a `uint32_t` count, or 0. If non-zero, the actual sequence count is `min(*sequenceCountAddress, maxSequenceCount)`. |
| `maxDrawCount` | Maximum draw count for `*_COUNT_*` tokens (0 if not used). |

### 9.2 vkCmdExecuteGeneratedCommandsEXT

```c
// Provided by VK_EXT_device_generated_commands
VKAPI_ATTR void VKAPI_CALL vkCmdExecuteGeneratedCommandsEXT(
    VkCommandBuffer                    commandBuffer,
    VkBool32                           isPreprocessed,
    const VkGeneratedCommandsInfoEXT*  pGeneratedCommandsInfo);
```

| Parameter | Description |
|---|---|
| `commandBuffer` | Command buffer into which the command is recorded. Must be a primary command buffer. |
| `isPreprocessed` | `VK_TRUE` if the data was already preprocessed via `vkCmdPreprocessGeneratedCommandsEXT`. `VK_FALSE` for implicit preprocessing. |
| `pGeneratedCommandsInfo` | Pointer to the generated commands info structure. |

**Key behaviors:**
- If `isPreprocessed == VK_FALSE`, preprocessing is performed implicitly and
  synchronized automatically.
- If `isPreprocessed == VK_TRUE`, the preprocess buffer must contain valid
  preprocessed data from a prior `vkCmdPreprocessGeneratedCommandsEXT` call with
  matching parameters.
- State affected by executed tokens becomes **undefined** after command execution.
- Processing for each sequence is **stateless** -- all state changes must occur before
  action commands within the sequence.

### 9.3 vkCmdPreprocessGeneratedCommandsEXT

```c
// Provided by VK_EXT_device_generated_commands
VKAPI_ATTR void VKAPI_CALL vkCmdPreprocessGeneratedCommandsEXT(
    VkCommandBuffer                    commandBuffer,
    const VkGeneratedCommandsInfoEXT*  pGeneratedCommandsInfo,
    VkCommandBuffer                    stateCommandBuffer);
```

| Parameter | Description |
|---|---|
| `commandBuffer` | Command buffer to record the preprocess command into. Can be on a different queue (e.g., async compute). |
| `pGeneratedCommandsInfo` | Same info struct that will be passed to `vkCmdExecuteGeneratedCommandsEXT`. |
| `stateCommandBuffer` | A **separate** command buffer that contains the rendering state (bound pipeline, descriptor sets, dynamic state, etc.) that will be active when `vkCmdExecuteGeneratedCommandsEXT` runs. The implementation reads state from this command buffer during preprocessing. |

**Key behaviors:**
- Executes in a separate logical pipeline (`VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT`).
- Requires **explicit synchronization** against the execution command.
- The indirect commands layout must have been created with
  `VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT`.

---

## 10. Synchronization

### 10.1 Pipeline Stage

```c
VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT = 0x00020000
```

This stage represents the preprocessing of device-generated commands. It executes in
a separate logical pipeline from graphics/compute.

### 10.2 Access Flags

```c
VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_EXT  = 0x00020000
VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_EXT = 0x00040000
```

### 10.3 Synchronization Patterns

**When using explicit preprocessing (separate preprocess step):**

```
// After compute shader fills indirect buffer:
srcStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT
dstStageMask  = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT
dstAccessMask = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_EXT

// Between preprocess and execute:
srcStageMask  = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT
srcAccessMask = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_EXT
dstStageMask  = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT
```

**When using implicit preprocessing (`isPreprocessed == VK_FALSE`):**

```
// Only need to synchronize the input buffer:
srcStageMask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT
dstStageMask  = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT
```

Implicit preprocessing automatically synchronizes internally.

---

## 11. Valid Usage Rules

### 11.1 Layout Creation Constraints

- `deviceGeneratedCommands` feature **must** be enabled.
- `tokenCount` must be > 0 and <= `maxIndirectCommandsTokenCount`.
- `indirectStride` must be <= `maxIndirectCommandsIndirectStride`.
- `shaderStages` must only contain stages in `supportedIndirectCommandsShaderStages`.
- Must contain **exactly one** action command token, and it **must be the last** token.
- At most **one** `EXECUTION_SET_EXT` token. If present, it **must be first**.
- At most **one** `SEQUENCE_INDEX_EXT` token.
- At most **one** `INDEX_BUFFER_EXT` token.
- `INDEX_BUFFER_EXT` token is only valid with indexed draw action tokens
  (`DRAW_INDEXED_EXT`, `DRAW_INDEXED_COUNT_EXT`).
- `VERTEX_BUFFER_EXT` token is only valid with non-mesh draw action tokens.
- Multiple `VERTEX_BUFFER_EXT` tokens require unique `vertexBindingUnit` values.
- Push constant ranges between tokens must **not overlap**.
- Token offsets must be **4-byte aligned**.
- Token offsets must be in **ascending order**.
- `PUSH_CONSTANT_EXT` and `SEQUENCE_INDEX_EXT` require `pipelineLayout` to be valid
  (unless `dynamicGeneratedPipelineLayout` is enabled).
- `DISPATCH_EXT` requires `VK_SHADER_STAGE_COMPUTE_BIT` in `shaderStages`.
- `TRACE_RAYS2_EXT` requires `rayTracingMaintenance1` feature.
- Mesh draw tokens require appropriate mesh shader stage bits.
- If `shaderStages` contains `VK_SHADER_STAGE_FRAGMENT_BIT`, it must also contain
  `VK_SHADER_STAGE_VERTEX_BIT` or `VK_SHADER_STAGE_MESH_BIT_EXT`.

### 11.2 Execution Constraints

- The command buffer must be a **primary** command buffer.
- `maxSequenceCount` must be <= `maxIndirectSequenceCount`.
- `preprocessSize` must be >= the size returned by
  `vkGetGeneratedCommandsMemoryRequirementsEXT`.
- If `isPreprocessed == VK_TRUE`, the preprocess buffer must contain valid data from a
  matching `vkCmdPreprocessGeneratedCommandsEXT` call.
- The indirect buffer must have `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT` or be a
  device-address-accessible buffer.
- All pipelines/shaders referenced by indices in the indirect buffer must be valid
  entries in the execution set.

### 11.3 Buffer Usage Requirements

- Buffers used as INDEX_BUFFER data must have `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`.
- Buffers used as VERTEX_BUFFER data must have `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`.
- Buffers used as draw/dispatch indirect data must have `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT`.
- Non-sparse buffers must be **completely bound** to a single `VkDeviceMemory`.

### 11.4 Execution Set Constraints

- Pipelines stored in the set must have `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`.
- Shaders stored in the set must have `VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT`.
- All pipelines/shaders in a set must be **layout-compatible** with the initial
  pipeline/shaders (same descriptor set layouts, push constant ranges, and shader stages).
- Execution set slots must not be modified while referenced by in-flight command buffers.

---

## 12. D3D12 Emulation Mappings

This section maps D3D12 Execute Indirect concepts to the Vulkan EXT equivalents, which
is particularly relevant for translation layers like DXVK and VKD3D-Proton.

### 12.1 Argument Structure Equivalents

| D3D12 Type | Vulkan Type |
|---|---|
| `D3D12_DRAW_ARGUMENTS` | `VkDrawIndirectCommand` |
| `D3D12_DRAW_INDEXED_ARGUMENTS` | `VkDrawIndexedIndirectCommand` |
| `D3D12_DISPATCH_ARGUMENTS` | `VkDispatchIndirectCommand` |
| `D3D12_INDEX_BUFFER_VIEW` | `VkBindIndexBufferIndirectCommandEXT` |
| `D3D12_VERTEX_BUFFER_VIEW` | `VkBindVertexBufferIndirectCommandEXT` |

### 12.2 Indirect Argument Type Mappings

| D3D12 Value | Vulkan Value |
|---|---|
| `D3D12_INDIRECT_ARGUMENT_TYPE_DRAW` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_VERTEX_BUFFER_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_INDEX_BUFFER_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_RAYS` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_TRACE_RAYS2_EXT` |
| `D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH` | `VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_EXT` |

**Note:** D3D12 CBV/SRV/UAV mappings all use `PUSH_CONSTANT_EXT` because the Vulkan
approach is to pass buffer device addresses as push constants (via
`GLSL_EXT_buffer_reference` / Vulkan 1.2 BDA).

---

## 13. VkStructureType Enum Values

```c
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT   = 1000572000
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT = 1000572001
VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT          = 1000572002
VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT                   = 1000572003
VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT                              = 1000572004
VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT                 = 1000572006
VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT                       = 1000572007
VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT                = 1000572008
VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_SHADER_EXT                  = 1000572009
VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT                 = 1000572010
VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_INFO_EXT                   = 1000572011
VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_LAYOUT_INFO_EXT            = 1000572012
VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT                     = 1000572013
VK_STRUCTURE_TYPE_GENERATED_COMMANDS_SHADER_INFO_EXT                       = 1000572014
```

---

## 14. Pipeline and Shader Creation Flags

### 14.1 Pipeline Flags

```c
// VkPipelineCreateFlagBits2KHR (64-bit)
VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT = 0x4000000000ULL
```

All pipelines stored in a pipeline-type `VkIndirectExecutionSetEXT` **must** have been
created with this flag.

### 14.2 Shader Flags

```c
// VkShaderCreateFlagBitsEXT
VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT = 0x00000080
```

All shaders stored in a shader-object-type `VkIndirectExecutionSetEXT` **must** have
been created with this flag.

---

## 15. Usage Workflow

### 15.1 Complete Setup Workflow

```
1. Create commands layout (VkIndirectCommandsLayoutEXT)
   - Define token sequence: e.g., [PUSH_CONSTANT, DISPATCH]
   - Set indirectStride = total bytes per sequence

2. (Optional) Create and populate an Indirect Execution Set
   - Create VkIndirectExecutionSetEXT with initial pipeline
   - Update with additional pipelines via vkUpdateIndirectExecutionSetPipelineEXT

3. Query memory requirements
   - Call vkGetGeneratedCommandsMemoryRequirementsEXT
   - Allocate DGC buffer and preprocess buffer

4. Fill DGC buffer
   - Use a compute shader to write per-sequence data
   - Or fill from host

5. Record and submit
   Option A (implicit preprocessing):
     vkCmdExecuteGeneratedCommandsEXT(cmdBuf, VK_FALSE, &info)

   Option B (explicit preprocessing for performance):
     // On async compute queue:
     vkCmdPreprocessGeneratedCommandsEXT(preprocessCmdBuf, &info, stateCmdBuf)
     // Submit preprocessCmdBuf first, synchronize, then:
     vkCmdExecuteGeneratedCommandsEXT(cmdBuf, VK_TRUE, &info)
```

### 15.2 Compute Dispatch Example (Conceptual)

```c
// Layout: push 12 bytes of constants, then dispatch
VkIndirectCommandsLayoutTokenEXT tokens[2];

// Token 0: Push constants
tokens[0].sType  = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
tokens[0].pNext  = NULL;
tokens[0].type   = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT;
tokens[0].offset = 0;
VkIndirectCommandsPushConstantTokenEXT pushToken = {
    .updateRange = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 12 }
};
tokens[0].data.pPushConstant = &pushToken;

// Token 1: Dispatch (action token, must be last)
tokens[1].sType  = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
tokens[1].pNext  = NULL;
tokens[1].type   = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
tokens[1].offset = 12;  // After push constant data
tokens[1].data   = {};  // No extra config for action tokens

VkIndirectCommandsLayoutCreateInfoEXT layoutInfo = {
    .sType          = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
    .flags          = 0,
    .shaderStages   = VK_SHADER_STAGE_COMPUTE_BIT,
    .indirectStride = 24,  // 12 bytes push constants + 12 bytes dispatch
    .pipelineLayout = myPipelineLayout,
    .tokenCount     = 2,
    .pTokens        = tokens,
};
vkCreateIndirectCommandsLayoutEXT(device, &layoutInfo, NULL, &layout);
```

Per-sequence buffer layout (24 bytes each):
```
Offset 0:  [float pushA] [float pushB] [float pushC]   (12 bytes)
Offset 12: [uint32 x]    [uint32 y]    [uint32 z]      (12 bytes)
```

### 15.3 Graphics Draw with Shader Switching Example (Conceptual)

```c
// Tokens: EXECUTION_SET -> INDEX_BUFFER -> VERTEX_BUFFER -> DRAW_INDEXED
// 4 tokens, stride = sizeof(uint32_t) + sizeof(VkBindIndexBufferIndirectCommandEXT)
//                   + sizeof(VkBindVertexBufferIndirectCommandEXT) + sizeof(VkDrawIndexedIndirectCommand)
//                 = 4 + 16 + 16 + 20 = 56 bytes per sequence

// Create execution set with initial pipeline:
VkIndirectExecutionSetPipelineInfoEXT pipeInfo = {
    .sType           = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT,
    .initialPipeline = myGraphicsPipeline,
    .maxPipelineCount = 64,
};
VkIndirectExecutionSetCreateInfoEXT setInfo = {
    .sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT,
    .type  = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT,
    .info.pPipelineInfo = &pipeInfo,
};
vkCreateIndirectExecutionSetEXT(device, &setInfo, NULL, &execSet);

// Register additional pipelines:
VkWriteIndirectExecutionSetPipelineEXT writes[] = {
    { ..., .index = 1, .pipeline = pipelineVariantA },
    { ..., .index = 2, .pipeline = pipelineVariantB },
};
vkUpdateIndirectExecutionSetPipelineEXT(device, execSet, 2, writes);
```

Per-sequence buffer layout (56 bytes each):
```
Offset 0:  [uint32 pipelineIndex]                          (4 bytes)
Offset 4:  [VkDeviceAddress ibAddr] [uint32 size] [VkIndexType type]  (16 bytes)
Offset 20: [VkDeviceAddress vbAddr] [uint32 size] [uint32 stride]     (16 bytes)
Offset 36: [uint32 indexCount] [uint32 instanceCount] [uint32 firstIndex]
           [int32 vertexOffset] [uint32 firstInstance]                 (20 bytes)
```

---

## 16. Unresolved Design Questions

From the proposal document, these issues remain open or are left for future extensions:

1. **Future command addition mechanism** -- New pointer members will be added to
   `VkIndirectCommandsTokenDataEXT` for new token types.

2. **Additional state inclusion** -- No additional state changes are permitted beyond
   the current token types, to enable fast and broad adoption. Future extensions may
   add more dynamic state.

3. **Shader stage and pipeline state mutability** -- Currently, only shaders can change
   between sequences. All other pipeline state must be identical.

4. **Potential merging with Shader Binding Tables** -- The indirect execution set concept
   overlaps with ray tracing SBTs and could potentially be unified.

5. **Additional alignment properties** -- May be needed for certain implementations.

6. **Index type value remapping** -- Unlike the NV extension, the EXT version does not
   support `pIndexTypeValues` remapping. Use `mode` flag instead.

7. **Indirect buffer reusability** -- Whether the same indirect buffer data can be
   reused across multiple executions (answer: yes, as long as the data remains valid).

8. **Sub-32-bit command data** -- Currently all data must be 4-byte aligned.

9. **Application data for preprocessing optimization** -- Applications could provide
   hints to improve preprocessing, but this is not currently supported.

---

## Appendix A: All New Object Types

- `VkIndirectExecutionSetEXT` (non-dispatchable handle)
- `VkIndirectCommandsLayoutEXT` (non-dispatchable handle)

## Appendix B: All New Commands

| Command | Description |
|---|---|
| `vkCreateIndirectExecutionSetEXT` | Create an indirect execution set |
| `vkDestroyIndirectExecutionSetEXT` | Destroy an indirect execution set |
| `vkUpdateIndirectExecutionSetPipelineEXT` | Update pipeline slots in a set |
| `vkUpdateIndirectExecutionSetShaderEXT` | Update shader slots in a set |
| `vkCreateIndirectCommandsLayoutEXT` | Create an indirect commands layout |
| `vkDestroyIndirectCommandsLayoutEXT` | Destroy an indirect commands layout |
| `vkGetGeneratedCommandsMemoryRequirementsEXT` | Query preprocess buffer requirements |
| `vkCmdPreprocessGeneratedCommandsEXT` | Preprocess generated commands (explicit) |
| `vkCmdExecuteGeneratedCommandsEXT` | Execute generated commands |

## Appendix C: All New Structures

| Structure | Description |
|---|---|
| `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT` | Feature query/enable |
| `VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT` | Property/limits query |
| `VkIndirectExecutionSetCreateInfoEXT` | Execution set creation info |
| `VkIndirectExecutionSetPipelineInfoEXT` | Pipeline-type set info |
| `VkIndirectExecutionSetShaderInfoEXT` | Shader-type set info |
| `VkIndirectExecutionSetShaderLayoutInfoEXT` | Per-shader layout info |
| `VkWriteIndirectExecutionSetPipelineEXT` | Pipeline slot update |
| `VkWriteIndirectExecutionSetShaderEXT` | Shader slot update |
| `VkIndirectCommandsLayoutCreateInfoEXT` | Layout creation info |
| `VkIndirectCommandsLayoutTokenEXT` | Token definition |
| `VkIndirectCommandsPushConstantTokenEXT` | Push constant token config |
| `VkIndirectCommandsIndexBufferTokenEXT` | Index buffer token config |
| `VkIndirectCommandsVertexBufferTokenEXT` | Vertex buffer token config |
| `VkIndirectCommandsExecutionSetTokenEXT` | Execution set token config |
| `VkBindIndexBufferIndirectCommandEXT` | Index buffer bind data (in DGC buffer) |
| `VkBindVertexBufferIndirectCommandEXT` | Vertex buffer bind data (in DGC buffer) |
| `VkDrawIndirectCountIndirectCommandEXT` | Multi-draw count data (in DGC buffer) |
| `VkGeneratedCommandsMemoryRequirementsInfoEXT` | Memory requirements query info |
| `VkGeneratedCommandsPipelineInfoEXT` | Pipeline hint for memory query |
| `VkGeneratedCommandsShaderInfoEXT` | Shader hint for memory query |
| `VkGeneratedCommandsInfoEXT` | Execution/preprocessing info |

## Appendix D: All New Unions

| Union | Description |
|---|---|
| `VkIndirectExecutionSetInfoEXT` | Selects pipeline or shader info for set creation |
| `VkIndirectCommandsTokenDataEXT` | Selects token-specific configuration data |

## Appendix E: All New Enums

| Enum | Values |
|---|---|
| `VkIndirectExecutionSetInfoTypeEXT` | `PIPELINES_EXT` (0), `SHADER_OBJECTS_EXT` (1) |
| `VkIndirectCommandsTokenTypeEXT` | `EXECUTION_SET_EXT` (0) through `DISPATCH_EXT` (9), plus mesh/trace extensions |
| `VkIndirectCommandsLayoutUsageFlagBitsEXT` | `EXPLICIT_PREPROCESS_BIT_EXT` (0x1), `UNORDERED_SEQUENCES_BIT_EXT` (0x2) |
| `VkIndirectCommandsInputModeFlagBitsEXT` | `VULKAN_INDEX_BUFFER_EXT` (0x1), `DXGI_INDEX_BUFFER_EXT` (0x2) |

## Appendix F: All New Enum Constants

Extending existing enums:

| Extended Enum | New Constant |
|---|---|
| `VkPipelineStageFlagBits` | `VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT` (0x00020000) |
| `VkAccessFlagBits` | `VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_EXT` (0x00020000) |
| `VkAccessFlagBits` | `VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_EXT` (0x00040000) |
| `VkObjectType` | `VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_EXT` |
| `VkObjectType` | `VK_OBJECT_TYPE_INDIRECT_EXECUTION_SET_EXT` |
| `VkPipelineCreateFlagBits2KHR` | `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT` (0x4000000000) |
| `VkShaderCreateFlagBitsEXT` | `VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT` (0x00000080) |
| `VkBufferUsageFlagBits2KHR` | (preprocess buffer usage, implementation-specific) |

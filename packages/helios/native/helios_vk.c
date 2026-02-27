/**
 * helios_vk.c — Minimal Vulkan compute bridge for Node.js via N-API.
 *
 * Written from scratch. No Vulkan SDK headers needed — we define the
 * minimum required structs/enums ourselves and load libvulkan.so via dlopen.
 *
 * Exposed to JS:
 *   initDevice()                          → { vendorId, deviceName, maxWorkgroupSize }
 *   createBuffer(byteLength)              → handle (number)
 *   uploadBuffer(handle, Float32Array)    → void
 *   readBuffer(handle, byteLength)        → Float32Array
 *   destroyBuffer(handle)                 → void
 *   createPipeline(spirvUint32Array)      → handle (number)
 *   dispatch(pipelineHandle, bufferHandles[], groupsX, groupsY, groupsZ) → void
 *   waitIdle()                            → void
 *   destroy()                             → void
 *
 * Build:
 *   gcc -shared -fPIC -o helios_vk.node helios_vk.c -ldl \
 *       -I$(node -e "console.log(require('path').join(process.execPath,'..','..','include','node'))")
 */

#define NAPI_VERSION 10
#include <node_api.h>

#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ── Vulkan type definitions (no SDK headers needed) ─────────────────────────

typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;

typedef enum { VK_SUCCESS = 0 } VkResult;
typedef enum {
  VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
  VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
  VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
  VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
  VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
  VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 15,
  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
  VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
  VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6,
  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001,
} VkStructureType;

typedef enum {
  VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001,
  VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002,
  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010,
  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020,
} VkBufferUsageFlagBits;

typedef enum {
  VK_SHARING_MODE_EXCLUSIVE = 0,
  VK_SHARING_MODE_CONCURRENT = 1,
} VkSharingMode;

typedef enum {
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001,
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002,
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004,
} VkMemoryPropertyFlagBits;

typedef enum {
  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
} VkDescriptorType;

typedef enum {
  VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020,
} VkShaderStageFlagBits;

typedef enum {
  VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
} VkCommandBufferLevel;

typedef enum {
  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001,
} VkCommandBufferUsageFlagBits;

typedef enum {
  VK_PIPELINE_BIND_POINT_COMPUTE = 1,
} VkPipelineBindPoint;

typedef enum {
  VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
  VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
} VkPhysicalDeviceType;

typedef enum {
  VK_QUEUE_GRAPHICS_BIT = 0x00000001,
  VK_QUEUE_COMPUTE_BIT = 0x00000002,
  VK_QUEUE_TRANSFER_BIT = 0x00000004,
} VkQueueFlagBits;

// Opaque handles (all pointers or uint64_t on 64-bit)
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkCommandPool;
typedef void* VkCommandBuffer;
typedef uint64_t VkBuffer;
typedef uint64_t VkDeviceMemory;
typedef uint64_t VkShaderModule;
typedef uint64_t VkPipelineLayout;
typedef uint64_t VkPipeline;
typedef uint64_t VkDescriptorSetLayout;
typedef uint64_t VkDescriptorPool;
typedef uint64_t VkDescriptorSet;
typedef uint64_t VkFence;
typedef uint64_t VkSemaphore;

// Structs
// Full VkPhysicalDeviceProperties is ~824 bytes. We define the fields we read
// and pad the rest so Vulkan doesn't write past our allocation.
typedef struct {
  uint32_t apiVersion;
  uint32_t driverVersion;
  uint32_t vendorID;
  uint32_t deviceID;
  uint32_t deviceType;
  char     deviceName[256];
  uint8_t  pipelineCacheUUID[16];
  uint8_t  _padding[1024]; // covers VkPhysicalDeviceLimits + VkPhysicalDeviceSparseProperties
} VkPhysicalDeviceProperties_partial;

typedef struct { VkFlags queueFlags; uint32_t queueCount; uint32_t timestampValidBits; uint32_t minImageTransferGranularity[3]; } VkQueueFamilyProperties;

typedef struct { uint32_t memoryTypeCount; struct { VkFlags propertyFlags; uint32_t heapIndex; } memoryTypes[32]; uint32_t memoryHeapCount; struct { VkDeviceSize size; VkFlags flags; } memoryHeaps[16]; } VkPhysicalDeviceMemoryProperties;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; VkDeviceSize size; VkFlags usage; VkSharingMode sharingMode; uint32_t queueFamilyIndexCount; const uint32_t* pQueueFamilyIndices; } VkBufferCreateInfo;

typedef struct { VkDeviceSize size; VkDeviceSize alignment; uint32_t memoryTypeBits; } VkMemoryRequirements;

typedef struct { VkStructureType sType; const void* pNext; VkDeviceSize allocationSize; uint32_t memoryTypeIndex; } VkMemoryAllocateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; size_t codeSize; const uint32_t* pCode; } VkShaderModuleCreateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t bindingCount; const void* pBindings; } VkDescriptorSetLayoutCreateInfo;

typedef struct { uint32_t binding; uint32_t descriptorType; uint32_t descriptorCount; VkFlags stageFlags; const void* pImmutableSamplers; } VkDescriptorSetLayoutBinding;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t setLayoutCount; const VkDescriptorSetLayout* pSetLayouts; uint32_t pushConstantRangeCount; const void* pPushConstantRanges; } VkPipelineLayoutCreateInfo;

typedef struct { VkFlags flags; VkShaderModule module; const char* pName; const void* pSpecializationInfo; VkStructureType sType; const void* pNext; uint32_t stage; } VkPipelineShaderStageCreateInfo_raw;

typedef struct {
  VkStructureType sType;
  const void* pNext;
  VkFlags flags;
  uint32_t stage;           // VkShaderStageFlagBits
  VkShaderModule module;
  const char* pName;
  const void* pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct {
  VkStructureType sType;
  const void* pNext;
  VkFlags flags;
  VkPipelineShaderStageCreateInfo stage;
  VkPipelineLayout layout;
  VkPipeline basePipelineHandle;
  int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t maxSets; uint32_t poolSizeCount; const void* pPoolSizes; } VkDescriptorPoolCreateInfo;

typedef struct { uint32_t type; uint32_t descriptorCount; } VkDescriptorPoolSize;

typedef struct { VkStructureType sType; const void* pNext; VkDescriptorPool descriptorPool; uint32_t descriptorSetCount; const VkDescriptorSetLayout* pSetLayouts; } VkDescriptorSetAllocateInfo;

typedef struct { VkBuffer buffer; VkDeviceSize offset; VkDeviceSize range; } VkDescriptorBufferInfo;

typedef struct { VkStructureType sType; const void* pNext; VkDescriptorSet dstSet; uint32_t dstBinding; uint32_t dstArrayElement; uint32_t descriptorCount; uint32_t descriptorType; const void* pImageInfo; const VkDescriptorBufferInfo* pBufferInfo; const void* pTexelBufferView; } VkWriteDescriptorSet;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t queueFamilyIndex; } VkCommandPoolCreateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkCommandPool commandPool; uint32_t level; uint32_t commandBufferCount; } VkCommandBufferAllocateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; const void* pInheritanceInfo; } VkCommandBufferBeginInfo;

typedef struct { VkStructureType sType; const void* pNext; uint32_t waitSemaphoreCount; const void* pWaitSemaphores; const void* pWaitDstStageMask; uint32_t commandBufferCount; const VkCommandBuffer* pCommandBuffers; uint32_t signalSemaphoreCount; const void* pSignalSemaphores; } VkSubmitInfo;

typedef struct { VkStructureType sType; const void* pNext; const char* pApplicationName; uint32_t applicationVersion; const char* pEngineName; uint32_t engineVersion; uint32_t apiVersion; } VkApplicationInfo;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; const VkApplicationInfo* pApplicationInfo; uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames; uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames; } VkInstanceCreateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t queueFamilyIndex; uint32_t queueCount; const float* pQueuePriorities; } VkDeviceQueueCreateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t queueCreateInfoCount; const VkDeviceQueueCreateInfo* pQueueCreateInfos; uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames; uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames; const void* pEnabledFeatures; } VkDeviceCreateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkDeviceMemory memory; VkDeviceSize offset; VkDeviceSize size; } VkMappedMemoryRange;

typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; } VkFenceCreateInfo;

typedef struct { VkStructureType sType; const void* pNext; VkFlags srcAccessMask; VkFlags dstAccessMask; } VkMemoryBarrier;

// VkBufferMemoryBarrier for per-buffer hazard tracking
// VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 44
typedef struct {
  VkStructureType sType;
  const void* pNext;
  VkFlags srcAccessMask;
  VkFlags dstAccessMask;
  uint32_t srcQueueFamilyIndex;
  uint32_t dstQueueFamilyIndex;
  VkBuffer buffer;
  VkDeviceSize offset;
  VkDeviceSize size;
} VkBufferMemoryBarrier;

// Timeline semaphore structs (Vulkan 1.2)
typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; } VkSemaphoreCreateInfo;
typedef struct { VkStructureType sType; const void* pNext; uint32_t semaphoreType; uint64_t initialValue; } VkSemaphoreTypeCreateInfo;
typedef struct { VkStructureType sType; const void* pNext; uint32_t waitSemaphoreValueCount; const uint64_t* pWaitSemaphoreValues; uint32_t signalSemaphoreValueCount; const uint64_t* pSignalSemaphoreValues; } VkTimelineSemaphoreSubmitInfo;
typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t semaphoreCount; const VkSemaphore* pSemaphores; const uint64_t* pValues; } VkSemaphoreWaitInfo;

// VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8
// VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46
// VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9
// VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO = 1000207002
// VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO = 1000207003
// VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO = 1000207004
#define VK_SEMAPHORE_TYPE_TIMELINE 1

// Pipeline stage / access flags
#define VK_PIPELINE_STAGE_TRANSFER_BIT         0x00001000
#define VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT   0x00000800
#define VK_ACCESS_SHADER_READ_BIT              0x00000020
#define VK_ACCESS_SHADER_WRITE_BIT             0x00000040
#define VK_ACCESS_TRANSFER_READ_BIT            0x00000800
#define VK_ACCESS_TRANSFER_WRITE_BIT           0x00001000

// Feature structs for f16 probing (Vulkan 1.2 core)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 1000059001
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES 49
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES 51

// Cooperative matrix (VK_KHR_cooperative_matrix)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR 1000506000
#define VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR 1000506002

// VkComponentTypeKHR values
#define VK_COMPONENT_TYPE_FLOAT16_KHR 0
#define VK_COMPONENT_TYPE_FLOAT32_KHR 1

// VkScopeKHR
#define VK_SCOPE_SUBGROUP_KHR 3

typedef struct {
  VkStructureType sType;
  void* pNext;
  VkBool32 cooperativeMatrix;
  VkBool32 cooperativeMatrixRobustBufferAccess;
} VkPhysicalDeviceCooperativeMatrixFeaturesKHR;

typedef struct {
  VkStructureType sType;
  void* pNext;
  uint32_t MSize;
  uint32_t NSize;
  uint32_t KSize;
  uint32_t AType;  // VkComponentTypeKHR
  uint32_t BType;
  uint32_t CType;
  uint32_t ResultType;
  VkBool32 saturatingAccumulation;
  uint32_t scope;  // VkScopeKHR
} VkCooperativeMatrixPropertiesKHR;

typedef struct { char extensionName[256]; uint32_t specVersion; } VkExtensionProperties;

// VkPhysicalDeviceVulkan11Features — storageBuffer16BitAccess is the first bool field
typedef struct {
  VkStructureType sType;
  void* pNext;
  VkBool32 storageBuffer16BitAccess;
  VkBool32 _rest[55];
} Vk11Features;

// VkPhysicalDeviceVulkan12Features — shaderFloat16 is field #8
typedef struct {
  VkStructureType sType;
  void* pNext;
  VkBool32 samplerMirrorClampToEdge;
  VkBool32 drawIndirectCount;
  VkBool32 storageBuffer8BitAccess;
  VkBool32 uniformAndStorageBuffer8BitAccess;
  VkBool32 storagePushConstant8;
  VkBool32 shaderBufferInt64Atomics;
  VkBool32 shaderSharedInt64Atomics;
  VkBool32 shaderFloat16;
  VkBool32 shaderInt8;
  VkBool32 _rest[160];
} Vk12Features;

typedef struct {
  VkStructureType sType;
  void* pNext;
  uint8_t features[220]; // VkPhysicalDeviceFeatures padding
} VkPhysicalDeviceFeatures2;

typedef void (*PFN_vkGetPhysicalDeviceFeatures2)(VkPhysicalDevice, VkPhysicalDeviceFeatures2*);

// Query pool (for GPU timestamps)
typedef uint64_t VkQueryPool;
typedef struct { VkStructureType sType; const void* pNext; VkFlags flags; uint32_t queryType; uint32_t queryCount; VkFlags pipelineStatistics; } VkQueryPoolCreateInfo;
#define VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO 11
#define VK_QUERY_TYPE_TIMESTAMP 2
#define VK_QUERY_RESULT_64_BIT 0x00000001
#define VK_QUERY_RESULT_WAIT_BIT 0x00000002
#define VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT 0x00000001
#define VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT 0x00002000

// ── Function pointer types ──────────────────────────────────────────────────

#define VK_WHOLE_SIZE (~0ULL)
#define VK_NULL_HANDLE 0

typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void     (*PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties_partial*);
typedef void     (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
typedef void     (*PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
typedef VkResult (*PFN_vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
typedef void     (*PFN_vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*);
typedef VkResult (*PFN_vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
typedef void     (*PFN_vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*);
typedef VkResult (*PFN_vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
typedef VkResult (*PFN_vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
typedef VkResult (*PFN_vkMapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
typedef void     (*PFN_vkUnmapMemory)(VkDevice, VkDeviceMemory);
typedef void     (*PFN_vkDestroyBuffer)(VkDevice, VkBuffer, const void*);
typedef void     (*PFN_vkFreeMemory)(VkDevice, VkDeviceMemory, const void*);
typedef VkResult (*PFN_vkCreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
typedef void     (*PFN_vkDestroyShaderModule)(VkDevice, VkShaderModule, const void*);
typedef VkResult (*PFN_vkCreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
typedef void     (*PFN_vkDestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void*);
typedef VkResult (*PFN_vkCreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
typedef void     (*PFN_vkDestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void*);
typedef VkResult (*PFN_vkCreateComputePipelines)(VkDevice, uint64_t, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*);
typedef void     (*PFN_vkDestroyPipeline)(VkDevice, VkPipeline, const void*);
typedef VkResult (*PFN_vkCreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
typedef void     (*PFN_vkDestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void*);
typedef VkResult (*PFN_vkAllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
typedef void     (*PFN_vkUpdateDescriptorSets)(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
typedef VkResult (*PFN_vkCreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
typedef void     (*PFN_vkDestroyCommandPool)(VkDevice, VkCommandPool, const void*);
typedef VkResult (*PFN_vkAllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
typedef VkResult (*PFN_vkBeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo*);
typedef void     (*PFN_vkCmdBindPipeline)(VkCommandBuffer, uint32_t, VkPipeline);
typedef void     (*PFN_vkCmdBindDescriptorSets)(VkCommandBuffer, uint32_t, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
typedef void     (*PFN_vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
typedef VkResult (*PFN_vkEndCommandBuffer)(VkCommandBuffer);
typedef VkResult (*PFN_vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, uint64_t);
typedef VkResult (*PFN_vkQueueWaitIdle)(VkQueue);
typedef VkResult (*PFN_vkDeviceWaitIdle)(VkDevice);
typedef void     (*PFN_vkDestroyDevice)(VkDevice, const void*);
typedef void     (*PFN_vkDestroyInstance)(VkInstance, const void*);
typedef VkResult (*PFN_vkResetCommandPool)(VkDevice, VkCommandPool, VkFlags);
typedef VkResult (*PFN_vkResetCommandBuffer)(VkCommandBuffer, VkFlags);
typedef VkResult (*PFN_vkResetDescriptorPool)(VkDevice, VkDescriptorPool, VkFlags);
typedef VkResult (*PFN_vkFlushMappedMemoryRanges)(VkDevice, uint32_t, const VkMappedMemoryRange*);
typedef VkResult (*PFN_vkInvalidateMappedMemoryRanges)(VkDevice, uint32_t, const VkMappedMemoryRange*);
typedef void     (*PFN_vkCmdCopyBuffer)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const void*);
typedef VkResult (*PFN_vkCreateFence)(VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
typedef void     (*PFN_vkDestroyFence)(VkDevice, VkFence, const void*);
typedef VkResult (*PFN_vkWaitForFences)(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);
typedef VkResult (*PFN_vkResetFences)(VkDevice, uint32_t, const VkFence*);
typedef void     (*PFN_vkCmdPipelineBarrier)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const void*, uint32_t, const void*);
typedef void     (*PFN_vkCmdPushConstants)(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void*);
typedef VkResult (*PFN_vkCreateSemaphore)(VkDevice, const VkSemaphoreCreateInfo*, const void*, VkSemaphore*);
typedef void     (*PFN_vkDestroySemaphore)(VkDevice, VkSemaphore, const void*);
typedef VkResult (*PFN_vkWaitSemaphores)(VkDevice, const VkSemaphoreWaitInfo*, uint64_t);
typedef VkResult (*PFN_vkGetSemaphoreCounterValue)(VkDevice, VkSemaphore, uint64_t*);
typedef VkResult (*PFN_vkCreateQueryPool)(VkDevice, const VkQueryPoolCreateInfo*, const void*, VkQueryPool*);
typedef void     (*PFN_vkDestroyQueryPool)(VkDevice, VkQueryPool, const void*);
typedef void     (*PFN_vkCmdWriteTimestamp)(VkCommandBuffer, VkFlags, VkQueryPool, uint32_t);
typedef void     (*PFN_vkCmdResetQueryPool)(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t);
typedef VkResult (*PFN_vkGetQueryPoolResults)(VkDevice, VkQueryPool, uint32_t, uint32_t, size_t, void*, VkDeviceSize, VkFlags);
typedef VkResult (*PFN_vkEnumerateDeviceExtensionProperties)(VkPhysicalDevice, const char*, uint32_t*, VkExtensionProperties*);
typedef VkResult (*PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)(VkPhysicalDevice, uint32_t*, VkCooperativeMatrixPropertiesKHR*);

// Push descriptors (VK_KHR_push_descriptor)
typedef void (*PFN_vkCmdPushDescriptorSetKHR)(VkCommandBuffer, uint32_t /*bindPoint*/, VkPipelineLayout, uint32_t /*set*/, uint32_t /*writeCount*/, const VkWriteDescriptorSet*);
#define VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR 0x00000001

// Push constant range
typedef struct { VkFlags stageFlags; uint32_t offset; uint32_t size; } VkPushConstantRange;

// Buffer copy region
typedef struct { VkDeviceSize srcOffset; VkDeviceSize dstOffset; VkDeviceSize size; } VkBufferCopy;

// ── Global state ────────────────────────────────────────────────────────────

static void* vk_lib = NULL;
static VkInstance instance = NULL;
static VkPhysicalDevice physDevice = NULL;
static VkDevice device = NULL;
static VkQueue computeQueue = NULL;
static VkCommandPool cmdPool = NULL;
static VkCommandBuffer dispatchCmdBuf = NULL;
static VkCommandBuffer transferCmdBuf = NULL;

// Multi-frame-in-flight ring: CPU records batch N+1 while GPU executes batch N
#define RING_SIZE 3
typedef struct {
  VkCommandPool   cmdPool;
  VkCommandBuffer cmd;
  VkDescriptorPool descPool;    // NULL if push descriptors available
  uint64_t        timelineValue;
} BatchCtx;
static BatchCtx g_ring[RING_SIZE];
static uint32_t g_ringHead = 0;

// Legacy single descriptor pool for single-dispatch path (dispatchCmdBuf)
static VkDescriptorPool singleDescPool = 0;
static VkFence persistentFence = 0;
static VkSemaphore timelineSem = 0;
static uint64_t nextTimelineValue = 1;
static uint64_t lastDispatchTimeline = 0;  // timeline value of last async dispatch
static int batchRecording = 0;
static uint32_t batchDispatchCount = 0;

// Per-buffer write tracking generation counter (arrays declared after MAX_BUFFERS)
static uint32_t bufWriteGeneration = 0;        // bumped on batchBegin

// Dispatch state cache — skip re-recording when pipeline+buffers+groups unchanged
static int dispatchCacheValid = 0;
static int32_t cachedPipeSlot = -1;
static int32_t cachedBufSlots[32];
static uint32_t cachedBufCount = 0;
static uint32_t cachedGX = 0, cachedGY = 0, cachedGZ = 0;
static uint8_t cachedPushData[128];
static uint32_t cachedPushSize = 0;
static uint32_t computeQueueFamily = 0;

// Legacy staging buffer for device-local transfers (fallback for large uploads)
static VkBuffer stagingBuffer = 0;
static VkDeviceMemory stagingMemory = 0;
static void* stagingMapped = NULL;
static VkDeviceSize stagingSize = 0;

// Async staging ring: 4 slots × 16MB each, persistently mapped
#define STAGING_RING_SIZE 4
#define STAGING_SLOT_BYTES (16 * 1024 * 1024)
typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  void* mapped;
  VkDeviceSize capacity;
  uint64_t timelineValue;  // 0 = free, >0 = in-flight until timeline reaches this
} StagingSlot;
static StagingSlot stagingRing[STAGING_RING_SIZE];
static int stagingRingInited = 0;
static VkPhysicalDeviceMemoryProperties memProps;
static char deviceNameStr[256] = {0};
static uint32_t vendorId = 0;
static VkQueryPool timestampPool = 0;
static int timestampsSupported = 0;
static float timestampPeriodNs = 1.0f;  // ns per tick (most GPUs = 1.0)
static int f16Supported = 0;            // can use f16 storage buffers
static PFN_vkGetPhysicalDeviceFeatures2 fp_vkGetPhysicalDeviceFeatures2;

// Push descriptors (VK_KHR_push_descriptor)
static int hasPushDescriptors = 0;
static PFN_vkCmdPushDescriptorSetKHR fp_vkCmdPushDescriptorSetKHR = NULL;

// Async transfer queue (separate DMA engine)
static VkQueue transferQueue = NULL;
static uint32_t transferQueueFamily = UINT32_MAX;
static VkCommandPool transferCmdPool = NULL;
static VkCommandBuffer xferCmdBuf = NULL;
static VkFence transferFence = 0;
static int hasAsyncTransfer = 0;

// Cooperative matrix state
static int coopMatSupported = 0;
static uint32_t coopMatM = 0, coopMatN = 0, coopMatK = 0;

// Function pointers
static PFN_vkCreateInstance                       fp_vkCreateInstance;
static PFN_vkEnumeratePhysicalDevices             fp_vkEnumeratePhysicalDevices;
static PFN_vkGetPhysicalDeviceProperties          fp_vkGetPhysicalDeviceProperties;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties fp_vkGetPhysicalDeviceQueueFamilyProperties;
static PFN_vkGetPhysicalDeviceMemoryProperties    fp_vkGetPhysicalDeviceMemoryProperties;
static PFN_vkCreateDevice                         fp_vkCreateDevice;
static PFN_vkGetDeviceQueue                       fp_vkGetDeviceQueue;
static PFN_vkCreateBuffer                         fp_vkCreateBuffer;
static PFN_vkGetBufferMemoryRequirements          fp_vkGetBufferMemoryRequirements;
static PFN_vkAllocateMemory                       fp_vkAllocateMemory;
static PFN_vkBindBufferMemory                     fp_vkBindBufferMemory;
static PFN_vkMapMemory                            fp_vkMapMemory;
static PFN_vkUnmapMemory                          fp_vkUnmapMemory;
static PFN_vkDestroyBuffer                        fp_vkDestroyBuffer;
static PFN_vkFreeMemory                           fp_vkFreeMemory;
static PFN_vkCreateShaderModule                   fp_vkCreateShaderModule;
static PFN_vkDestroyShaderModule                  fp_vkDestroyShaderModule;
static PFN_vkCreateDescriptorSetLayout            fp_vkCreateDescriptorSetLayout;
static PFN_vkDestroyDescriptorSetLayout           fp_vkDestroyDescriptorSetLayout;
static PFN_vkCreatePipelineLayout                 fp_vkCreatePipelineLayout;
static PFN_vkDestroyPipelineLayout                fp_vkDestroyPipelineLayout;
static PFN_vkCreateComputePipelines               fp_vkCreateComputePipelines;
static PFN_vkDestroyPipeline                      fp_vkDestroyPipeline;
static PFN_vkCreateDescriptorPool                 fp_vkCreateDescriptorPool;
static PFN_vkDestroyDescriptorPool                fp_vkDestroyDescriptorPool;
static PFN_vkAllocateDescriptorSets               fp_vkAllocateDescriptorSets;
static PFN_vkUpdateDescriptorSets                 fp_vkUpdateDescriptorSets;
static PFN_vkCreateCommandPool                    fp_vkCreateCommandPool;
static PFN_vkDestroyCommandPool                   fp_vkDestroyCommandPool;
static PFN_vkAllocateCommandBuffers               fp_vkAllocateCommandBuffers;
static PFN_vkBeginCommandBuffer                   fp_vkBeginCommandBuffer;
static PFN_vkCmdBindPipeline                      fp_vkCmdBindPipeline;
static PFN_vkCmdBindDescriptorSets                fp_vkCmdBindDescriptorSets;
static PFN_vkCmdDispatch                          fp_vkCmdDispatch;
static PFN_vkEndCommandBuffer                     fp_vkEndCommandBuffer;
static PFN_vkQueueSubmit                          fp_vkQueueSubmit;
static PFN_vkQueueWaitIdle                        fp_vkQueueWaitIdle;
static PFN_vkDeviceWaitIdle                       fp_vkDeviceWaitIdle;
static PFN_vkDestroyDevice                        fp_vkDestroyDevice;
static PFN_vkDestroyInstance                      fp_vkDestroyInstance;
static PFN_vkResetCommandPool                     fp_vkResetCommandPool;
static PFN_vkResetCommandBuffer                   fp_vkResetCommandBuffer;
static PFN_vkResetDescriptorPool                  fp_vkResetDescriptorPool;
static PFN_vkFlushMappedMemoryRanges              fp_vkFlushMappedMemoryRanges;
static PFN_vkInvalidateMappedMemoryRanges         fp_vkInvalidateMappedMemoryRanges;
static PFN_vkCmdCopyBuffer                        fp_vkCmdCopyBuffer;
static PFN_vkCreateFence                          fp_vkCreateFence;
static PFN_vkDestroyFence                         fp_vkDestroyFence;
static PFN_vkWaitForFences                        fp_vkWaitForFences;
static PFN_vkResetFences                          fp_vkResetFences;
static PFN_vkCmdPipelineBarrier                   fp_vkCmdPipelineBarrier;
static PFN_vkCmdPushConstants                     fp_vkCmdPushConstants;
static PFN_vkCreateSemaphore                      fp_vkCreateSemaphore;
static PFN_vkDestroySemaphore                     fp_vkDestroySemaphore;
static PFN_vkWaitSemaphores                       fp_vkWaitSemaphores;
static PFN_vkGetSemaphoreCounterValue             fp_vkGetSemaphoreCounterValue;
static PFN_vkCreateQueryPool                      fp_vkCreateQueryPool;
static PFN_vkDestroyQueryPool                     fp_vkDestroyQueryPool;
static PFN_vkCmdWriteTimestamp                    fp_vkCmdWriteTimestamp;
static PFN_vkCmdResetQueryPool                    fp_vkCmdResetQueryPool;
static PFN_vkGetQueryPoolResults                  fp_vkGetQueryPoolResults;

// ── Memory sub-allocator (slab allocator) ────────────────────────────────────

#define SLAB_INITIAL_SIZE   (64 * 1024 * 1024)   // 64 MB per slab
#define SLAB_MAX_SIZE       (1024 * 1024 * 1024)  // 1 GB max per slab
#define MAX_SLABS           64

typedef struct {
  VkDeviceMemory memory;
  VkDeviceSize   capacity;
  VkDeviceSize   head;          // bump pointer
  void*          mapped;        // persistent mapping (NULL if not host-visible)
  uint32_t       memoryTypeIdx;
  uint32_t       refCount;      // number of live buffer allocations from this slab
} Slab;

typedef struct {
  Slab     slabs[MAX_SLABS];
  uint32_t slabCount;
  uint32_t memoryTypeIdx;       // preferred memory type for this pool
  int      hostVisible;
} SlabPool;

static SlabPool devicePool = {0};  // device-local
static SlabPool hostPool = {0};    // host-visible + coherent
static VkDeviceSize slabAlignment = 256;  // queried at init from a probe buffer

static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// Allocate from a slab pool. Returns memory + offset, or fails.
typedef struct { VkDeviceMemory memory; VkDeviceSize offset; void* mappedBase; } SlabAlloc;

static int slabPoolAlloc(SlabPool* pool, VkDeviceSize size, SlabAlloc* out) {
  VkDeviceSize aligned_size = alignUp(size, slabAlignment);

  // Try existing slabs
  for (uint32_t i = 0; i < pool->slabCount; i++) {
    Slab* s = &pool->slabs[i];
    VkDeviceSize offset = alignUp(s->head, slabAlignment);
    if (offset + aligned_size <= s->capacity) {
      s->head = offset + aligned_size;
      s->refCount++;
      out->memory = s->memory;
      out->offset = offset;
      out->mappedBase = s->mapped;
      return 1;
    }
  }

  // Need a new slab
  if (pool->slabCount >= MAX_SLABS) return 0;

  VkDeviceSize slabSize = SLAB_INITIAL_SIZE;
  // Make slab big enough for this allocation
  while (slabSize < aligned_size) slabSize <<= 1;
  // Each subsequent slab doubles (up to max)
  for (uint32_t i = 0; i < pool->slabCount && slabSize < SLAB_MAX_SIZE; i++) slabSize <<= 1;
  if (slabSize > SLAB_MAX_SIZE) slabSize = SLAB_MAX_SIZE;
  if (slabSize < aligned_size) slabSize = aligned_size;

  VkMemoryAllocateInfo allocInfo = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = slabSize,
    .memoryTypeIndex = pool->memoryTypeIdx,
  };
  Slab* s = &pool->slabs[pool->slabCount];
  VkResult res = fp_vkAllocateMemory(device, &allocInfo, NULL, &s->memory);
  if (res != VK_SUCCESS) return 0;

  s->capacity = slabSize;
  s->head = aligned_size;  // first alloc starts at 0
  s->memoryTypeIdx = pool->memoryTypeIdx;
  s->mapped = NULL;
  s->refCount = 1;

  if (pool->hostVisible) {
    fp_vkMapMemory(device, s->memory, 0, slabSize, 0, &s->mapped);
  }

  pool->slabCount++;

  out->memory = s->memory;
  out->offset = 0;
  out->mappedBase = s->mapped;
  return 1;
}

// ── Resource tracking ───────────────────────────────────────────────────────

#define MAX_BUFFERS   8192
#define MAX_PIPELINES 256

typedef struct {
  VkBuffer       buffer;
  VkDeviceMemory memory;       // slab memory (shared, NOT owned by this slot)
  VkDeviceSize   memOffset;    // offset within slab
  VkDeviceSize   size;
  void*          mapped;       // persistent mapping (NULL if device-local only)
  int            hostVisible;  // 1 if host-visible
  int            active;
  int            slabAllocated; // 1 if from slab allocator (don't vkFreeMemory)
  uint64_t       lastWriteTimeline;  // timeline value when last dispatch wrote to this
} BufferSlot;

typedef struct {
  VkPipeline            pipeline;
  VkPipelineLayout      layout;
  VkDescriptorSetLayout descLayout;
  uint32_t              numBindings;
  uint32_t              pushConstantSize;
  int                   active;
} PipelineSlot;

static BufferSlot   buffers[MAX_BUFFERS];
static PipelineSlot pipelines[MAX_PIPELINES];

// Per-buffer write tracking for fine-grained barriers within a batch (O(1) lookup)
static uint32_t bufWriteDispatch[MAX_BUFFERS]; // dispatch index of last write per buffer slot
static uint32_t bufWriteGen[MAX_BUFFERS];      // generation stamp per buffer slot

// ── Helpers ─────────────────────────────────────────────────────────────────

#define LOAD_VK(name) do { \
  fp_##name = (PFN_##name)dlsym(vk_lib, #name); \
  if (!fp_##name) { napi_throw_error(env, NULL, "Failed to load " #name); return NULL; } \
} while(0)

static uint32_t findMemoryType(uint32_t typeFilter, VkFlags properties) {
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  return UINT32_MAX;
}

static int allocBufferSlot(void) {
  for (int i = 0; i < MAX_BUFFERS; i++) {
    if (!buffers[i].active) return i;
  }
  return -1;
}

static int allocPipelineSlot(void) {
  for (int i = 0; i < MAX_PIPELINES; i++) {
    if (!pipelines[i].active) return i;
  }
  return -1;
}

static VkResult ensureStagingBuffer(VkDeviceSize needed) {
  if (stagingSize >= needed) return VK_SUCCESS;

  // Destroy old staging buffer
  if (stagingBuffer != 0) {
    fp_vkUnmapMemory(device, stagingMemory);
    fp_vkDestroyBuffer(device, stagingBuffer, NULL);
    fp_vkFreeMemory(device, stagingMemory, NULL);
    stagingBuffer = 0; stagingMemory = 0; stagingMapped = NULL; stagingSize = 0;
  }

  // Round up to next power of 2, minimum 1MB
  VkDeviceSize newSize = 1 << 20;
  while (newSize < needed) newSize <<= 1;

  uint32_t stageFamilies[2] = {computeQueueFamily, transferQueueFamily};
  VkBufferCreateInfo bufInfo = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = newSize,
    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    .sharingMode = hasAsyncTransfer ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = hasAsyncTransfer ? 2 : 0,
    .pQueueFamilyIndices = hasAsyncTransfer ? stageFamilies : NULL,
  };
  VkResult res = fp_vkCreateBuffer(device, &bufInfo, NULL, &stagingBuffer);
  if (res != VK_SUCCESS) return res;

  VkMemoryRequirements memReq;
  fp_vkGetBufferMemoryRequirements(device, stagingBuffer, &memReq);

  VkFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t memType = findMemoryType(memReq.memoryTypeBits, memFlags);
  if (memType == UINT32_MAX) return (VkResult)-1;

  VkMemoryAllocateInfo allocInfo = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = memReq.size,
    .memoryTypeIndex = memType,
  };
  res = fp_vkAllocateMemory(device, &allocInfo, NULL, &stagingMemory);
  if (res != VK_SUCCESS) return res;

  fp_vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);
  fp_vkMapMemory(device, stagingMemory, 0, newSize, 0, &stagingMapped);
  stagingSize = newSize;
  return VK_SUCCESS;
}

// Initialize the async staging ring (called lazily on first device-local upload)
static int initStagingRing(void) {
  if (stagingRingInited) return 1;
  uint32_t stageFamilies[2] = {computeQueueFamily, transferQueueFamily};
  for (int i = 0; i < STAGING_RING_SIZE; i++) {
    VkBufferCreateInfo bi = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = STAGING_SLOT_BYTES,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = hasAsyncTransfer ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = hasAsyncTransfer ? 2 : 0,
      .pQueueFamilyIndices = hasAsyncTransfer ? stageFamilies : NULL,
    };
    VkResult r = fp_vkCreateBuffer(device, &bi, NULL, &stagingRing[i].buffer);
    if (r != VK_SUCCESS) return 0;
    VkMemoryRequirements req;
    fp_vkGetBufferMemoryRequirements(device, stagingRing[i].buffer, &req);
    VkFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t mt = findMemoryType(req.memoryTypeBits, memFlags);
    if (mt == UINT32_MAX) return 0;
    VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = mt,
    };
    r = fp_vkAllocateMemory(device, &ai, NULL, &stagingRing[i].memory);
    if (r != VK_SUCCESS) return 0;
    fp_vkBindBufferMemory(device, stagingRing[i].buffer, stagingRing[i].memory, 0);
    fp_vkMapMemory(device, stagingRing[i].memory, 0, STAGING_SLOT_BYTES, 0, &stagingRing[i].mapped);
    stagingRing[i].capacity = STAGING_SLOT_BYTES;
    stagingRing[i].timelineValue = 0;
  }
  stagingRingInited = 1;
  return 1;
}

// Submit a command buffer with fence-based sync (used for transfers)
static VkResult submitCmdBufSync(VkCommandBuffer cmdBuf) {
  VkSubmitInfo submitInfo = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmdBuf,
  };
  VkResult res = fp_vkQueueSubmit(computeQueue, 1, &submitInfo, persistentFence);
  if (res != VK_SUCCESS) return res;
  res = fp_vkWaitForFences(device, 1, &persistentFence, 1, ~0ULL);
  fp_vkResetFences(device, 1, &persistentFence);
  return res;
}

// Submit a command buffer with timeline semaphore signal (async — no host wait)
// Returns signalValue on success, 0 on failure (0 is never a valid timeline signal).
static uint64_t submitCmdBufAsync(VkCommandBuffer cmdBuf) {
  uint64_t signalValue = nextTimelineValue; // tentative — only commit on success
  VkTimelineSemaphoreSubmitInfo tsInfo = {
    .sType = 1000207003, // VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO
    .signalSemaphoreValueCount = 1,
    .pSignalSemaphoreValues = &signalValue,
  };
  uint32_t waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkSubmitInfo submitInfo = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .pNext = &tsInfo,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmdBuf,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &timelineSem,
  };
  VkResult res = fp_vkQueueSubmit(computeQueue, 1, &submitInfo, 0);
  if (res != VK_SUCCESS) {
    // Don't advance timeline — the semaphore was never signaled.
    return 0;
  }
  nextTimelineValue = signalValue + 1; // commit — advance past the value we just signaled
  return signalValue;
}

// Wait until timeline semaphore reaches the given value
static void waitTimelineValue(uint64_t value) {
  if (value == 0) return;
  uint64_t completed;
  fp_vkGetSemaphoreCounterValue(device, timelineSem, &completed);
  if (completed >= value) return;
  VkSemaphoreWaitInfo waitInfo = {
    .sType = 1000207004, // VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO
    .semaphoreCount = 1,
    .pSemaphores = &timelineSem,
    .pValues = &value,
  };
  fp_vkWaitSemaphores(device, &waitInfo, ~0ULL);
}

// ── N-API: initDevice() ─────────────────────────────────────────────────────

static napi_value napi_initDevice(napi_env env, napi_callback_info info) {
  // Load libvulkan.so
  vk_lib = dlopen("libvulkan.so.1", RTLD_NOW);
  if (!vk_lib) vk_lib = dlopen("libvulkan.so", RTLD_NOW);
  if (!vk_lib) {
    napi_throw_error(env, NULL, "Failed to load libvulkan.so — is Vulkan installed?");
    return NULL;
  }

  // Load all function pointers
  LOAD_VK(vkCreateInstance);
  LOAD_VK(vkEnumeratePhysicalDevices);
  LOAD_VK(vkGetPhysicalDeviceProperties);
  LOAD_VK(vkGetPhysicalDeviceQueueFamilyProperties);
  LOAD_VK(vkGetPhysicalDeviceMemoryProperties);
  LOAD_VK(vkCreateDevice);
  LOAD_VK(vkGetDeviceQueue);
  LOAD_VK(vkCreateBuffer);
  LOAD_VK(vkGetBufferMemoryRequirements);
  LOAD_VK(vkAllocateMemory);
  LOAD_VK(vkBindBufferMemory);
  LOAD_VK(vkMapMemory);
  LOAD_VK(vkUnmapMemory);
  LOAD_VK(vkDestroyBuffer);
  LOAD_VK(vkFreeMemory);
  LOAD_VK(vkCreateShaderModule);
  LOAD_VK(vkDestroyShaderModule);
  LOAD_VK(vkCreateDescriptorSetLayout);
  LOAD_VK(vkDestroyDescriptorSetLayout);
  LOAD_VK(vkCreatePipelineLayout);
  LOAD_VK(vkDestroyPipelineLayout);
  LOAD_VK(vkCreateComputePipelines);
  LOAD_VK(vkDestroyPipeline);
  LOAD_VK(vkCreateDescriptorPool);
  LOAD_VK(vkDestroyDescriptorPool);
  LOAD_VK(vkAllocateDescriptorSets);
  LOAD_VK(vkUpdateDescriptorSets);
  LOAD_VK(vkCreateCommandPool);
  LOAD_VK(vkDestroyCommandPool);
  LOAD_VK(vkAllocateCommandBuffers);
  LOAD_VK(vkBeginCommandBuffer);
  LOAD_VK(vkCmdBindPipeline);
  LOAD_VK(vkCmdBindDescriptorSets);
  LOAD_VK(vkCmdDispatch);
  LOAD_VK(vkEndCommandBuffer);
  LOAD_VK(vkQueueSubmit);
  LOAD_VK(vkQueueWaitIdle);
  LOAD_VK(vkDeviceWaitIdle);
  LOAD_VK(vkDestroyDevice);
  LOAD_VK(vkDestroyInstance);
  LOAD_VK(vkResetCommandPool);
  LOAD_VK(vkResetCommandBuffer);
  LOAD_VK(vkResetDescriptorPool);
  LOAD_VK(vkFlushMappedMemoryRanges);
  LOAD_VK(vkInvalidateMappedMemoryRanges);
  LOAD_VK(vkCmdCopyBuffer);
  LOAD_VK(vkCreateFence);
  LOAD_VK(vkDestroyFence);
  LOAD_VK(vkWaitForFences);
  LOAD_VK(vkResetFences);
  LOAD_VK(vkCmdPipelineBarrier);
  LOAD_VK(vkCmdPushConstants);
  LOAD_VK(vkCreateSemaphore);
  LOAD_VK(vkDestroySemaphore);
  LOAD_VK(vkWaitSemaphores);
  LOAD_VK(vkGetSemaphoreCounterValue);
  LOAD_VK(vkCreateQueryPool);
  LOAD_VK(vkDestroyQueryPool);
  LOAD_VK(vkCmdWriteTimestamp);
  LOAD_VK(vkCmdResetQueryPool);
  LOAD_VK(vkGetQueryPoolResults);

  // Optional: load vkGetPhysicalDeviceFeatures2 (for f16 probing)
  fp_vkGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)dlsym(vk_lib, "vkGetPhysicalDeviceFeatures2");

  // Optional: load extension enumeration (for cooperative matrix probing)
  PFN_vkEnumerateDeviceExtensionProperties fp_vkEnumerateDeviceExtensionProperties =
    (PFN_vkEnumerateDeviceExtensionProperties)dlsym(vk_lib, "vkEnumerateDeviceExtensionProperties");

  // Create instance
  VkApplicationInfo appInfo = {
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName = "Helios",
    .applicationVersion = 1,
    .pEngineName = "Helios",
    .engineVersion = 1,
    .apiVersion = (1 << 22) | (2 << 12), // VK_API_VERSION_1_2
  };
  VkInstanceCreateInfo createInfo = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo = &appInfo,
  };
  VkResult res = fp_vkCreateInstance(&createInfo, NULL, &instance);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateInstance failed");
    return NULL;
  }

  // Pick physical device (prefer discrete GPU, fall back to integrated)
  uint32_t devCount = 0;
  fp_vkEnumeratePhysicalDevices(instance, &devCount, NULL);
  if (devCount == 0) {
    napi_throw_error(env, NULL, "No Vulkan-capable GPU found");
    return NULL;
  }
  VkPhysicalDevice* devs = malloc(sizeof(VkPhysicalDevice) * devCount);
  fp_vkEnumeratePhysicalDevices(instance, &devCount, devs);

  physDevice = devs[0]; // default to first
  VkPhysicalDeviceProperties_partial bestProps;
  fp_vkGetPhysicalDeviceProperties(devs[0], &bestProps);

  for (uint32_t i = 0; i < devCount; i++) {
    VkPhysicalDeviceProperties_partial props;
    fp_vkGetPhysicalDeviceProperties(devs[i], &props);
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      physDevice = devs[i];
      bestProps = props;
      break;
    }
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && bestProps.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      physDevice = devs[i];
      bestProps = props;
    }
  }
  free(devs);

  strncpy(deviceNameStr, bestProps.deviceName, 255);
  vendorId = bestProps.vendorID;

  // Get memory properties
  fp_vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);

  // Find compute queue family
  uint32_t qfCount = 0;
  fp_vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qfCount, NULL);
  VkQueueFamilyProperties* qfProps = malloc(sizeof(VkQueueFamilyProperties) * qfCount);
  fp_vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &qfCount, qfProps);

  computeQueueFamily = UINT32_MAX;
  uint32_t timestampValidBits = 0;
  for (uint32_t i = 0; i < qfCount; i++) {
    if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      computeQueueFamily = i;
      timestampValidBits = qfProps[i].timestampValidBits;
      break;
    }
  }

  // Find dedicated transfer queue family (separate DMA engine for async transfers)
  transferQueueFamily = UINT32_MAX;
  for (uint32_t i = 0; i < qfCount; i++) {
    if (i == computeQueueFamily) continue;
    if (qfProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
      // Prefer a family that's transfer-only (dedicated DMA, no compute/graphics)
      if (!(qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          !(qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
        transferQueueFamily = i;
        break;
      }
      if (transferQueueFamily == UINT32_MAX) transferQueueFamily = i;
    }
  }
  free(qfProps);

  if (computeQueueFamily == UINT32_MAX) {
    napi_throw_error(env, NULL, "No compute queue family found");
    return NULL;
  }

  // Probe f16 support (Vulkan 1.2: shaderFloat16 + storageBuffer16BitAccess)
  f16Supported = 0;
  Vk11Features feat11 = {0};
  Vk12Features feat12 = {0};
  if (fp_vkGetPhysicalDeviceFeatures2) {
    feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feat11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    feat11.pNext = &feat12;
    VkPhysicalDeviceFeatures2 features2 = {0};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &feat11;
    fp_vkGetPhysicalDeviceFeatures2(physDevice, &features2);
    if (feat12.shaderFloat16 && feat11.storageBuffer16BitAccess) {
      f16Supported = 1;
    }
  }

  // Probe device extensions (cooperative matrix, push descriptors)
  coopMatSupported = 0;
  int hasCoopMatExt = 0;
  hasPushDescriptors = 0;
  if (fp_vkEnumerateDeviceExtensionProperties) {
    uint32_t extCount = 0;
    fp_vkEnumerateDeviceExtensionProperties(physDevice, NULL, &extCount, NULL);
    if (extCount > 0) {
      VkExtensionProperties* exts = (VkExtensionProperties*)malloc(sizeof(VkExtensionProperties) * extCount);
      fp_vkEnumerateDeviceExtensionProperties(physDevice, NULL, &extCount, exts);
      for (uint32_t i = 0; i < extCount; i++) {
        if (strcmp(exts[i].extensionName, "VK_KHR_cooperative_matrix") == 0) {
          hasCoopMatExt = 1;
        }
        if (strcmp(exts[i].extensionName, "VK_KHR_push_descriptor") == 0) {
          hasPushDescriptors = 1;
        }
      }
      free(exts);
    }
  }

  // If extension present, probe the feature
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures = {0};
  if (hasCoopMatExt && fp_vkGetPhysicalDeviceFeatures2) {
    coopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    // Re-probe features with coop matrix chained in
    Vk12Features feat12b = {0};
    feat12b.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feat12b.pNext = &coopMatFeatures;
    Vk11Features feat11b = {0};
    feat11b.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    feat11b.pNext = &feat12b;
    VkPhysicalDeviceFeatures2 features2b = {0};
    features2b.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2b.pNext = &feat11b;
    fp_vkGetPhysicalDeviceFeatures2(physDevice, &features2b);
    if (coopMatFeatures.cooperativeMatrix) {
      hasCoopMatExt = 2; // confirmed feature support
    } else {
      hasCoopMatExt = 0;
    }
  }

  // Create logical device (with f16 features if supported)
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queueCreates[2];
  uint32_t queueCreateCount = 1;
  queueCreates[0] = (VkDeviceQueueCreateInfo){
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = computeQueueFamily,
    .queueCount = 1,
    .pQueuePriorities = &priority,
  };
  if (transferQueueFamily != UINT32_MAX) {
    queueCreates[queueCreateCount++] = (VkDeviceQueueCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = transferQueueFamily,
      .queueCount = 1,
      .pQueuePriorities = &priority,
    };
  }

  // Enable f16 features in pNext chain
  Vk11Features enableFeat11 = {0};
  Vk12Features enableFeat12 = {0};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR enableCoopMat = {0};
  void* devicePNext = NULL;
  if (f16Supported) {
    enableFeat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enableFeat12.shaderFloat16 = 1;
    enableFeat11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    enableFeat11.storageBuffer16BitAccess = 1;
    enableFeat11.pNext = &enableFeat12;
    devicePNext = &enableFeat11;
  }

  // Enable push descriptors if supported
  const char* enabledExtensions[4];
  uint32_t enabledExtCount = 0;
  if (hasPushDescriptors) {
    enabledExtensions[enabledExtCount++] = "VK_KHR_push_descriptor";
  }

  // Chain cooperative matrix features if supported
  if (hasCoopMatExt == 2) {
    enableCoopMat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    enableCoopMat.cooperativeMatrix = 1;
    // Chain into pNext
    if (devicePNext) {
      enableFeat12.pNext = &enableCoopMat;
    } else if (f16Supported) {
      enableFeat11.pNext = &enableFeat12;
      enableFeat12.pNext = &enableCoopMat;
      devicePNext = &enableFeat11;
    } else {
      devicePNext = &enableCoopMat;
    }
    enabledExtensions[enabledExtCount++] = "VK_KHR_cooperative_matrix";
  }

  VkDeviceCreateInfo devCreate = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = devicePNext,
    .queueCreateInfoCount = queueCreateCount,
    .pQueueCreateInfos = queueCreates,
    .enabledExtensionCount = enabledExtCount,
    .ppEnabledExtensionNames = enabledExtCount > 0 ? enabledExtensions : NULL,
  };
  res = fp_vkCreateDevice(physDevice, &devCreate, NULL, &device);
  if (res != VK_SUCCESS) {
    // Retry without optional features
    f16Supported = 0;
    hasCoopMatExt = 0;
    hasPushDescriptors = 0;
    devCreate.pNext = NULL;
    devCreate.enabledExtensionCount = 0;
    devCreate.ppEnabledExtensionNames = NULL;
    res = fp_vkCreateDevice(physDevice, &devCreate, NULL, &device);
    if (res != VK_SUCCESS) {
      napi_throw_error(env, NULL, "vkCreateDevice failed");
      return NULL;
    }
  }

  fp_vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

  // Load push descriptor function pointer if extension was enabled
  if (hasPushDescriptors) {
    typedef void* (*PFN_vkGetDeviceProcAddr)(VkDevice, const char*);
    PFN_vkGetDeviceProcAddr fp_getDevProcAddr = (PFN_vkGetDeviceProcAddr)dlsym(vk_lib, "vkGetDeviceProcAddr");
    if (fp_getDevProcAddr) {
      fp_vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)fp_getDevProcAddr(device, "vkCmdPushDescriptorSetKHR");
      if (!fp_vkCmdPushDescriptorSetKHR) hasPushDescriptors = 0;
    } else {
      hasPushDescriptors = 0;
    }
  }

  // Get transfer queue if separate family found
  hasAsyncTransfer = 0;
  if (transferQueueFamily != UINT32_MAX) {
    fp_vkGetDeviceQueue(device, transferQueueFamily, 0, &transferQueue);
    hasAsyncTransfer = 1;
  }

  // Query cooperative matrix properties to find supported tile sizes
  if (hasCoopMatExt == 2) {
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR fp_getCoopMatProps =
      (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)dlsym(vk_lib,
        "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (fp_getCoopMatProps) {
      uint32_t propCount = 0;
      fp_getCoopMatProps(physDevice, &propCount, NULL);
      if (propCount > 0) {
        VkCooperativeMatrixPropertiesKHR* props =
          (VkCooperativeMatrixPropertiesKHR*)malloc(sizeof(VkCooperativeMatrixPropertiesKHR) * propCount);
        for (uint32_t i = 0; i < propCount; i++) {
          props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
          props[i].pNext = NULL;
        }
        fp_getCoopMatProps(physDevice, &propCount, props);
        // Find a combination with f16 A/B, f32 C/Result, subgroup scope
        for (uint32_t i = 0; i < propCount; i++) {
          if (props[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
              props[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
              props[i].CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
              props[i].ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
              props[i].scope == VK_SCOPE_SUBGROUP_KHR) {
            coopMatSupported = 1;
            coopMatM = props[i].MSize;
            coopMatN = props[i].NSize;
            coopMatK = props[i].KSize;
            break;
          }
        }
        free(props);
      }
    }
  }

  // Create command pool for single-dispatch and transfer paths
  VkCommandPoolCreateInfo poolInfo = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = 0x00000002, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    .queueFamilyIndex = computeQueueFamily,
  };
  res = fp_vkCreateCommandPool(device, &poolInfo, NULL, &cmdPool);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateCommandPool failed");
    return NULL;
  }

  // Create small descriptor pool for single-dispatch path
  VkDescriptorPoolSize singlePoolSize = {
    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = 256,
  };
  VkDescriptorPoolCreateInfo singleDpInfo = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 32,
    .poolSizeCount = 1,
    .pPoolSizes = &singlePoolSize,
  };
  res = fp_vkCreateDescriptorPool(device, &singleDpInfo, NULL, &singleDescPool);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateDescriptorPool (single) failed");
    return NULL;
  }

  // Create multi-frame-in-flight ring: 3 slots with independent cmdPool + descPool
  memset(g_ring, 0, sizeof(g_ring));
  g_ringHead = 0;
  for (int ri = 0; ri < RING_SIZE; ri++) {
    VkCommandPoolCreateInfo ringPoolInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = 0x00000002, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
      .queueFamilyIndex = computeQueueFamily,
    };
    res = fp_vkCreateCommandPool(device, &ringPoolInfo, NULL, &g_ring[ri].cmdPool);
    if (res != VK_SUCCESS) {
      napi_throw_error(env, NULL, "vkCreateCommandPool (ring) failed");
      return NULL;
    }
    VkCommandBufferAllocateInfo ringCbInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = g_ring[ri].cmdPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
    };
    res = fp_vkAllocateCommandBuffers(device, &ringCbInfo, &g_ring[ri].cmd);
    if (res != VK_SUCCESS) {
      napi_throw_error(env, NULL, "vkAllocateCommandBuffers (ring) failed");
      return NULL;
    }
    // Only create descriptor pool if push descriptors not available
    if (!hasPushDescriptors) {
      VkDescriptorPoolSize ringDescSize = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 16384,
      };
      VkDescriptorPoolCreateInfo ringDpInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2048,
        .poolSizeCount = 1,
        .pPoolSizes = &ringDescSize,
      };
      res = fp_vkCreateDescriptorPool(device, &ringDpInfo, NULL, &g_ring[ri].descPool);
      if (res != VK_SUCCESS) {
        napi_throw_error(env, NULL, "vkCreateDescriptorPool (ring) failed");
        return NULL;
      }
    }
    g_ring[ri].timelineValue = 0;
  }

  // Pre-allocate command buffers for single-dispatch and transfer paths
  VkCommandBuffer cmdBufs[2];
  VkCommandBufferAllocateInfo persistCbInfo = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = cmdPool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 2,
  };
  res = fp_vkAllocateCommandBuffers(device, &persistCbInfo, cmdBufs);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkAllocateCommandBuffers failed");
    return NULL;
  }
  dispatchCmdBuf = cmdBufs[0];
  transferCmdBuf = cmdBufs[1];

  // Create persistent fence for synchronization (used for transfers)
  VkFenceCreateInfo fenceInfo = { .sType = 8 /* VK_STRUCTURE_TYPE_FENCE_CREATE_INFO */ };
  res = fp_vkCreateFence(device, &fenceInfo, NULL, &persistentFence);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateFence failed");
    return NULL;
  }

  // Create timeline semaphore for async dispatch (Vulkan 1.2)
  VkSemaphoreTypeCreateInfo semTypeInfo = {
    .sType = 1000207002, // VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO
    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
    .initialValue = 0,
  };
  VkSemaphoreCreateInfo semInfo = {
    .sType = 9, // VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    .pNext = &semTypeInfo,
  };
  res = fp_vkCreateSemaphore(device, &semInfo, NULL, &timelineSem);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateSemaphore (timeline) failed");
    return NULL;
  }
  nextTimelineValue = 1;
  lastDispatchTimeline = 0;

  // Create transfer command pool + buffer if separate transfer queue available
  if (hasAsyncTransfer) {
    VkCommandPoolCreateInfo xferPoolInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = 0x00000002, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
      .queueFamilyIndex = transferQueueFamily,
    };
    res = fp_vkCreateCommandPool(device, &xferPoolInfo, NULL, &transferCmdPool);
    if (res == VK_SUCCESS) {
      VkCommandBufferAllocateInfo xferCbInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = transferCmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
      };
      fp_vkAllocateCommandBuffers(device, &xferCbInfo, &xferCmdBuf);
      VkFenceCreateInfo xferFenceInfo = { .sType = 8 /* VK_STRUCTURE_TYPE_FENCE_CREATE_INFO */ };
      fp_vkCreateFence(device, &xferFenceInfo, NULL, &transferFence);
    } else {
      hasAsyncTransfer = 0;
      transferQueue = NULL;
    }
  }

  // Create timestamp query pool if supported
  timestampsSupported = (timestampValidBits > 0) ? 1 : 0;
  if (timestampsSupported) {
    VkQueryPoolCreateInfo qpInfo = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 2,  // start + end for one dispatch
    };
    res = fp_vkCreateQueryPool(device, &qpInfo, NULL, &timestampPool);
    if (res != VK_SUCCESS) {
      timestampsSupported = 0;
      timestampPool = 0;
    }
  }

  // Extract timestampPeriod from device properties (float at known offset)
  // VkPhysicalDeviceLimits starts at offset 292 in VkPhysicalDeviceProperties.
  // timestampPeriod is at offset 276 within VkPhysicalDeviceLimits.
  // Total offset from start of struct: 292 + 276 = 568 bytes.
  // We access it as raw bytes from our padded struct.
  {
    uint8_t fullProps[1024];
    memset(fullProps, 0, sizeof(fullProps));
    fp_vkGetPhysicalDeviceProperties(physDevice, (VkPhysicalDeviceProperties_partial*)fullProps);
    float period;
    memcpy(&period, fullProps + 568, sizeof(float));
    if (period > 0.0f) timestampPeriodNs = period;
  }

  // Init resource slots
  memset(buffers, 0, sizeof(buffers));
  memset(pipelines, 0, sizeof(pipelines));

  // Probe alignment for slab allocator: create a test storage buffer
  {
    VkBufferCreateInfo probeInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 1024,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer probeBuffer;
    res = fp_vkCreateBuffer(device, &probeInfo, NULL, &probeBuffer);
    if (res == VK_SUCCESS) {
      VkMemoryRequirements probeReq;
      fp_vkGetBufferMemoryRequirements(device, probeBuffer, &probeReq);
      slabAlignment = probeReq.alignment;
      if (slabAlignment < 256) slabAlignment = 256;  // minimum safety
      fp_vkDestroyBuffer(device, probeBuffer, NULL);
    }
  }

  // Initialize slab pools
  memset(&devicePool, 0, sizeof(devicePool));
  memset(&hostPool, 0, sizeof(hostPool));
  {
    // Find memory types for each pool
    // Device-local
    uint32_t devType = findMemoryType(0xFFFFFFFF, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (devType == UINT32_MAX) devType = findMemoryType(0xFFFFFFFF, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    devicePool.memoryTypeIdx = devType;
    devicePool.hostVisible = 0;
    // Check if device-local is also host-visible (integrated GPU)
    if (devType != UINT32_MAX && (memProps.memoryTypes[devType].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      devicePool.hostVisible = 1;
    }

    // Host-visible + coherent
    uint32_t hostType = findMemoryType(0xFFFFFFFF, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    hostPool.memoryTypeIdx = hostType;
    hostPool.hostVisible = 1;
  }

  // Return device info
  napi_value result, val;
  napi_create_object(env, &result);

  napi_create_string_utf8(env, deviceNameStr, strlen(deviceNameStr), &val);
  napi_set_named_property(env, result, "deviceName", val);

  napi_create_uint32(env, vendorId, &val);
  napi_set_named_property(env, result, "vendorId", val);

  napi_get_boolean(env, f16Supported, &val);
  napi_set_named_property(env, result, "f16Supported", val);

  napi_get_boolean(env, hasAsyncTransfer, &val);
  napi_set_named_property(env, result, "hasAsyncTransfer", val);

  napi_get_boolean(env, coopMatSupported, &val);
  napi_set_named_property(env, result, "coopMatSupported", val);

  napi_create_uint32(env, coopMatM, &val);
  napi_set_named_property(env, result, "coopMatM", val);

  napi_create_uint32(env, coopMatN, &val);
  napi_set_named_property(env, result, "coopMatN", val);

  napi_create_uint32(env, coopMatK, &val);
  napi_set_named_property(env, result, "coopMatK", val);

  napi_get_boolean(env, hasPushDescriptors, &val);
  napi_set_named_property(env, result, "hasPushDescriptors", val);

  return result;
}

// ── N-API: createBuffer(byteLength, hostVisible) ────────────────────────────

static napi_value napi_createBuffer(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  uint32_t byteLength;
  napi_get_value_uint32(env, args[0], &byteLength);

  int32_t hostVisible = 1;
  if (argc > 1) napi_get_value_int32(env, args[1], &hostVisible);

  int slot = allocBufferSlot();
  if (slot < 0) {
    napi_throw_error(env, NULL, "Max buffers reached");
    return NULL;
  }

  uint32_t sharedFamilies[2] = {computeQueueFamily, transferQueueFamily};
  VkBufferCreateInfo bufInfo = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = byteLength,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    .sharingMode = hasAsyncTransfer ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = hasAsyncTransfer ? 2 : 0,
    .pQueueFamilyIndices = hasAsyncTransfer ? sharedFamilies : NULL,
  };

  VkResult res = fp_vkCreateBuffer(device, &bufInfo, NULL, &buffers[slot].buffer);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateBuffer failed");
    return NULL;
  }

  VkMemoryRequirements memReq;
  fp_vkGetBufferMemoryRequirements(device, buffers[slot].buffer, &memReq);

  // Determine memory type
  // hostVisible=0 → use device pool (DEVICE_LOCAL, HBM3 on discrete GPUs)
  // hostVisible=1 → use host pool (HOST_VISIBLE, system RAM for staging)
  // On discrete GPUs, device-local memory is NOT host-visible; upload/readback
  // uses staging buffers. On integrated GPUs, the device pool IS host-visible
  // (detected in initDevice and flagged in devicePool.hostVisible).
  int useHostPool = hostVisible;
  if (!hostVisible && devicePool.hostVisible) {
    // Integrated GPU or device pool is already host-visible — can map directly
    hostVisible = 1;
  }

  SlabPool* pool = useHostPool ? &hostPool : &devicePool;
  SlabAlloc salloc;
  // Slab allocation for device-local buffers is DISABLED: persistent buffers
  // (model params) share slabs with temporary buffers (intermediates), preventing
  // slab space reclamation even with refcounting. The JS-side buffer pool handles
  // buffer recycling; individual vkAllocateMemory/vkFreeMemory properly returns
  // memory to the driver. Host-visible buffers still use slab (staging is long-lived).
  int slabCompatible = useHostPool
    ? (memReq.memoryTypeBits & (1u << pool->memoryTypeIdx)) != 0
    : 0;
  if (slabCompatible && slabPoolAlloc(pool, memReq.size, &salloc)) {
    // Slab allocation succeeded
    res = fp_vkBindBufferMemory(device, buffers[slot].buffer, salloc.memory, salloc.offset);
    if (res != VK_SUCCESS) {
      napi_throw_error(env, NULL, "vkBindBufferMemory (slab) failed");
      return NULL;
    }
    buffers[slot].memory = salloc.memory;
    buffers[slot].memOffset = salloc.offset;
    buffers[slot].size = byteLength;
    buffers[slot].hostVisible = useHostPool;
    buffers[slot].slabAllocated = 1;
    buffers[slot].mapped = salloc.mappedBase ? (char*)salloc.mappedBase + salloc.offset : NULL;
  } else {
    // Fallback: individual allocation
    VkFlags memFlags;
    uint32_t memType;
    if (hostVisible) {
      memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      memType = findMemoryType(memReq.memoryTypeBits, memFlags);
    } else {
      memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      memType = findMemoryType(memReq.memoryTypeBits, memFlags);
    }
    if (memType == UINT32_MAX) {
      memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      memType = findMemoryType(memReq.memoryTypeBits, memFlags);
      hostVisible = 1;
    }
    if (memType == UINT32_MAX) {
      napi_throw_error(env, NULL, "No suitable memory type found");
      return NULL;
    }

    VkMemoryAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memReq.size,
      .memoryTypeIndex = memType,
    };
    VkDeviceMemory mem;
    res = fp_vkAllocateMemory(device, &allocInfo, NULL, &mem);
    if (res != VK_SUCCESS) {
      napi_throw_error(env, NULL, "vkAllocateMemory failed");
      return NULL;
    }
    fp_vkBindBufferMemory(device, buffers[slot].buffer, mem, 0);
    buffers[slot].memory = mem;
    buffers[slot].memOffset = 0;
    buffers[slot].size = byteLength;
    buffers[slot].hostVisible = hostVisible;
    buffers[slot].slabAllocated = 0;
    buffers[slot].mapped = NULL;
    if (hostVisible) {
      fp_vkMapMemory(device, mem, 0, byteLength, 0, &buffers[slot].mapped);
    }
  }
  buffers[slot].active = 1;

  napi_value result;
  napi_create_int32(env, slot, &result);
  return result;
}

// ── N-API: uploadBuffer(handle, Float32Array) ───────────────────────────────

static napi_value napi_uploadBuffer(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t slot;
  napi_get_value_int32(env, args[0], &slot);

  void* data;
  size_t byteLen;
  napi_get_typedarray_info(env, args[1], NULL, NULL, &data, NULL, NULL);
  napi_get_typedarray_info(env, args[1], NULL, &byteLen, NULL, NULL, NULL);
  // byteLen is element count, need byte length
  size_t byteLength = byteLen * 4; // Float32Array = 4 bytes per element

  if (slot < 0 || slot >= MAX_BUFFERS || !buffers[slot].active) {
    napi_throw_error(env, NULL, "Invalid buffer handle");
    return NULL;
  }

  size_t copyLen = byteLength < buffers[slot].size ? byteLength : buffers[slot].size;

  if (buffers[slot].mapped) {
    // Host-visible: direct memcpy (still need to wait if GPU is writing to this buffer)
    waitTimelineValue(buffers[slot].lastWriteTimeline);
    memcpy(buffers[slot].mapped, data, copyLen);
  } else if (copyLen <= STAGING_SLOT_BYTES && initStagingRing()) {
    // Device-local + fits in a staging ring slot: async copy (no blocking wait)
    // Find a free staging slot
    uint64_t completed;
    fp_vkGetSemaphoreCounterValue(device, timelineSem, &completed);
    int found = -1;
    for (int si = 0; si < STAGING_RING_SIZE; si++) {
      if (stagingRing[si].timelineValue == 0 || stagingRing[si].timelineValue <= completed) {
        found = si;
        break;
      }
    }
    if (found < 0) {
      // All staging slots in flight — wait for the oldest one
      uint64_t oldest = stagingRing[0].timelineValue;
      found = 0;
      for (int si = 1; si < STAGING_RING_SIZE; si++) {
        if (stagingRing[si].timelineValue < oldest) {
          oldest = stagingRing[si].timelineValue;
          found = si;
        }
      }
      waitTimelineValue(stagingRing[found].timelineValue);
    }

    memcpy(stagingRing[found].mapped, data, copyLen);

    // If a batch is currently recording, record the copy into the batch command buffer
    if (batchRecording) {
      VkBufferCopy region = { 0, 0, copyLen };
      fp_vkCmdCopyBuffer(g_ring[g_ringHead].cmd, stagingRing[found].buffer, buffers[slot].buffer, 1, &region);
      // Add a transfer→compute barrier so subsequent dispatches see the copy
      VkBufferMemoryBarrier xferBarrier = {
        .sType = 44,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = 0xFFFFFFFF,
        .dstQueueFamilyIndex = 0xFFFFFFFF,
        .buffer = buffers[slot].buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
      };
      fp_vkCmdPipelineBarrier(g_ring[g_ringHead].cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &xferBarrier, 0, NULL);
      // Staging slot will be marked with the batch's timeline value in batchSubmit
      stagingRing[found].timelineValue = nextTimelineValue; // tentative — will be committed on submit
    } else {
      // No batch recording — submit a standalone async copy
      fp_vkResetCommandBuffer(transferCmdBuf, 0);
      VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
      fp_vkBeginCommandBuffer(transferCmdBuf, &bi);
      VkBufferCopy region = { 0, 0, copyLen };
      fp_vkCmdCopyBuffer(transferCmdBuf, stagingRing[found].buffer, buffers[slot].buffer, 1, &region);
      fp_vkEndCommandBuffer(transferCmdBuf);
      uint64_t tv = submitCmdBufAsync(transferCmdBuf);
      stagingRing[found].timelineValue = tv;
      if (tv > 0) lastDispatchTimeline = tv;
    }
  } else {
    // Device-local, large buffer: blocking staging copy (fallback)
    waitTimelineValue(lastDispatchTimeline);
    VkResult r = ensureStagingBuffer(copyLen);
    if (r != VK_SUCCESS) { napi_throw_error(env, NULL, "staging buffer alloc failed"); return NULL; }
    memcpy(stagingMapped, data, copyLen);

    VkCommandBuffer cb = hasAsyncTransfer ? xferCmdBuf : transferCmdBuf;
    VkQueue q = hasAsyncTransfer ? transferQueue : computeQueue;
    VkFence f = hasAsyncTransfer ? transferFence : persistentFence;

    fp_vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    fp_vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region = { 0, 0, copyLen };
    fp_vkCmdCopyBuffer(cb, stagingBuffer, buffers[slot].buffer, 1, &region);
    fp_vkEndCommandBuffer(cb);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    r = fp_vkQueueSubmit(q, 1, &si, f);
    if (r != VK_SUCCESS) { napi_throw_error(env, NULL, "upload staging submit failed"); return NULL; }
    fp_vkWaitForFences(device, 1, &f, 1, ~0ULL);
    fp_vkResetFences(device, 1, &f);
  }

  return NULL;
}

// ── N-API: readBuffer(handle) → Float32Array ────────────────────────────────

static napi_value napi_readBuffer(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t slot;
  napi_get_value_int32(env, args[0], &slot);

  if (slot < 0 || slot >= MAX_BUFFERS || !buffers[slot].active) {
    napi_throw_error(env, NULL, "Invalid buffer handle");
    return NULL;
  }

  // Wait for any in-flight dispatch that wrote to this buffer.
  // For mapped (host-visible) buffers, also wait for lastDispatchTimeline
  // because batchDispatch doesn't track per-buffer write timelines.
  uint64_t waitFor = buffers[slot].lastWriteTimeline;
  if (buffers[slot].mapped && lastDispatchTimeline > waitFor) {
    waitFor = lastDispatchTimeline;
  }
  if (waitFor > 0) {
    waitTimelineValue(waitFor);
  }

  uint32_t elemCount = (uint32_t)(buffers[slot].size / 4);
  napi_value arrayBuf, result;
  void* outData;
  napi_create_arraybuffer(env, buffers[slot].size, &outData, &arrayBuf);

  if (buffers[slot].mapped) {
    // Host-visible: direct memcpy
    memcpy(outData, buffers[slot].mapped, buffers[slot].size);
  } else {
    // Device-local: copy to staging, then read
    VkResult r = ensureStagingBuffer(buffers[slot].size);
    if (r != VK_SUCCESS) { napi_throw_error(env, NULL, "staging buffer alloc failed"); return NULL; }

    // Use dedicated transfer queue if available (frees compute queue)
    VkCommandBuffer cb = hasAsyncTransfer ? xferCmdBuf : transferCmdBuf;
    VkQueue q = hasAsyncTransfer ? transferQueue : computeQueue;
    VkFence f = hasAsyncTransfer ? transferFence : persistentFence;

    fp_vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    fp_vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region = { 0, 0, buffers[slot].size };
    fp_vkCmdCopyBuffer(cb, buffers[slot].buffer, stagingBuffer, 1, &region);
    fp_vkEndCommandBuffer(cb);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    r = fp_vkQueueSubmit(q, 1, &si, f);
    if (r != VK_SUCCESS) { napi_throw_error(env, NULL, "readback staging submit failed"); return NULL; }
    fp_vkWaitForFences(device, 1, &f, 1, ~0ULL);
    fp_vkResetFences(device, 1, &f);
    memcpy(outData, stagingMapped, buffers[slot].size);
  }

  napi_create_typedarray(env, napi_float32_array, elemCount, arrayBuf, 0, &result);
  return result;
}

// ── N-API: destroyBuffer(handle) ────────────────────────────────────────────

static napi_value napi_destroyBuffer(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t slot;
  napi_get_value_int32(env, args[0], &slot);

  if (slot >= 0 && slot < MAX_BUFFERS && buffers[slot].active) {
    fp_vkDestroyBuffer(device, buffers[slot].buffer, NULL);
    if (!buffers[slot].slabAllocated) {
      // Only free memory if it was individually allocated (not from slab)
      if (buffers[slot].mapped) {
        fp_vkUnmapMemory(device, buffers[slot].memory);
      }
      fp_vkFreeMemory(device, buffers[slot].memory, NULL);
    } else {
      // Slab-allocated: decrement ref count. When a slab reaches zero refs,
      // reset its bump pointer so the space can be reused for new allocations.
      // This prevents unbounded slab growth during training where buffers are
      // constantly created and destroyed through the JS-side buffer pool.
      VkDeviceMemory slabMem = buffers[slot].memory;
      SlabPool* pools[2] = { &devicePool, &hostPool };
      for (int p = 0; p < 2; p++) {
        for (uint32_t si = 0; si < pools[p]->slabCount; si++) {
          if (pools[p]->slabs[si].memory == slabMem) {
            if (pools[p]->slabs[si].refCount > 0) {
              pools[p]->slabs[si].refCount--;
              if (pools[p]->slabs[si].refCount == 0) {
                pools[p]->slabs[si].head = 0;  // reset bump pointer — all space reclaimed
              }
            }
            goto slab_found;
          }
        }
      }
      slab_found:;
    }
    buffers[slot].mapped = NULL;
    buffers[slot].active = 0;
  }

  return NULL;
}

// ── N-API: createPipeline(spirvUint32Array, numBindings) → handle ───────────

static napi_value napi_createPipeline(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  void* spirvData;
  size_t spirvLen; // element count (uint32)
  napi_get_typedarray_info(env, args[0], NULL, &spirvLen, &spirvData, NULL, NULL);

  uint32_t numBindings = 4;
  if (argc > 1) napi_get_value_uint32(env, args[1], &numBindings);

  uint32_t pushConstantSize = 0;
  if (argc > 2) napi_get_value_uint32(env, args[2], &pushConstantSize);

  int slot = allocPipelineSlot();
  if (slot < 0) {
    napi_throw_error(env, NULL, "Max pipelines reached");
    return NULL;
  }

  // Create shader module
  VkShaderModuleCreateInfo shaderInfo = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = spirvLen * 4,
    .pCode = (const uint32_t*)spirvData,
  };
  VkShaderModule shaderModule;
  VkResult res = fp_vkCreateShaderModule(device, &shaderInfo, NULL, &shaderModule);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateShaderModule failed");
    return NULL;
  }

  // Create descriptor set layout (N storage buffer bindings)
  VkDescriptorSetLayoutBinding* bindings = malloc(sizeof(VkDescriptorSetLayoutBinding) * numBindings);
  for (uint32_t i = 0; i < numBindings; i++) {
    bindings[i] = (VkDescriptorSetLayoutBinding){
      .binding = i,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .pImmutableSamplers = NULL,
    };
  }
  VkDescriptorSetLayoutCreateInfo descLayoutInfo = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .flags = hasPushDescriptors ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
    .bindingCount = numBindings,
    .pBindings = bindings,
  };
  res = fp_vkCreateDescriptorSetLayout(device, &descLayoutInfo, NULL, &pipelines[slot].descLayout);
  free(bindings);
  if (res != VK_SUCCESS) {
    fp_vkDestroyShaderModule(device, shaderModule, NULL);
    napi_throw_error(env, NULL, "vkCreateDescriptorSetLayout failed");
    return NULL;
  }

  // Create pipeline layout (with optional push constant range)
  VkPushConstantRange pushRange;
  VkPipelineLayoutCreateInfo layoutInfo = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &pipelines[slot].descLayout,
  };
  if (pushConstantSize > 0) {
    pushRange = (VkPushConstantRange){
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = pushConstantSize,
    };
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
  }
  res = fp_vkCreatePipelineLayout(device, &layoutInfo, NULL, &pipelines[slot].layout);
  if (res != VK_SUCCESS) {
    fp_vkDestroyShaderModule(device, shaderModule, NULL);
    napi_throw_error(env, NULL, "vkCreatePipelineLayout failed");
    return NULL;
  }

  // Create compute pipeline
  // VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
  VkComputePipelineCreateInfo pipeInfo = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = 18,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shaderModule,
      .pName = "main",
    },
    .layout = pipelines[slot].layout,
  };

  res = fp_vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, NULL, &pipelines[slot].pipeline);
  fp_vkDestroyShaderModule(device, shaderModule, NULL);

  if (res != VK_SUCCESS) {
    char errbuf[128];
    snprintf(errbuf, sizeof(errbuf), "vkCreateComputePipelines failed (VkResult=%d, spirv_words=%zu, bindings=%u)", (int)res, spirvLen, numBindings);
    napi_throw_error(env, NULL, errbuf);
    return NULL;
  }

  pipelines[slot].numBindings = numBindings;
  pipelines[slot].pushConstantSize = pushConstantSize;
  pipelines[slot].active = 1;

  napi_value result;
  napi_create_int32(env, slot, &result);
  return result;
}

// ── N-API: dispatch(pipelineHandle, bufferHandles[], gX, gY, gZ, pushConstants?) ──

static napi_value napi_dispatch(napi_env env, napi_callback_info info) {
  size_t argc = 6;
  napi_value args[6];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t pipeSlot;
  napi_get_value_int32(env, args[0], &pipeSlot);

  if (pipeSlot < 0 || pipeSlot >= MAX_PIPELINES || !pipelines[pipeSlot].active) {
    napi_throw_error(env, NULL, "Invalid pipeline handle");
    return NULL;
  }

  // Get buffer handles array
  uint32_t bufCount;
  napi_get_array_length(env, args[1], &bufCount);

  if (bufCount == 0 || bufCount > 32) {
    napi_throw_range_error(env, NULL, "dispatch: bufCount must be 1..32");
    return NULL;
  }

  int32_t bufSlots[32];
  for (uint32_t i = 0; i < bufCount; i++) {
    napi_value elem;
    napi_get_element(env, args[1], i, &elem);
    napi_get_value_int32(env, elem, &bufSlots[i]);
  }

  uint32_t gX = 1, gY = 1, gZ = 1;
  napi_get_value_uint32(env, args[2], &gX);
  if (argc > 3) napi_get_value_uint32(env, args[3], &gY);
  if (argc > 4) napi_get_value_uint32(env, args[4], &gZ);

  // Optional push constants (arg 5 = Float32Array)
  void* pushData = NULL;
  uint32_t pushSize = pipelines[pipeSlot].pushConstantSize;
  if (argc > 5 && pushSize > 0) {
    napi_get_typedarray_info(env, args[5], NULL, NULL, &pushData, NULL, NULL);
  } else {
    pushSize = 0;
  }

  // Check dispatch cache — skip re-recording when pipeline+buffers+groups+pushdata unchanged
  int cacheHit = dispatchCacheValid &&
                 pipeSlot == cachedPipeSlot &&
                 bufCount == cachedBufCount &&
                 gX == cachedGX && gY == cachedGY && gZ == cachedGZ &&
                 pushSize == cachedPushSize &&
                 memcmp(bufSlots, cachedBufSlots, bufCount * sizeof(int32_t)) == 0 &&
                 (pushSize == 0 || memcmp(pushData, cachedPushData, pushSize) == 0);

  if (!cacheHit) {
    PipelineSlot* ps = &pipelines[pipeSlot];

    // Must wait for any in-flight dispatch to finish before re-recording cmd buffer
    waitTimelineValue(lastDispatchTimeline);

    // Reset persistent descriptor pool (frees all prior sets)
    fp_vkResetDescriptorPool(device, singleDescPool, 0);

    // Allocate descriptor set from persistent pool
    VkDescriptorSetAllocateInfo dsAllocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = singleDescPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &ps->descLayout,
    };
    VkDescriptorSet descSet;
    VkResult res = fp_vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet);
    if (res != VK_SUCCESS) {
      napi_throw_error(env, NULL, "vkAllocateDescriptorSets failed");
      return NULL;
    }

    // Write descriptors (stack arrays — no malloc)
    VkDescriptorBufferInfo bufInfos[32];
    VkWriteDescriptorSet writes[32];
    for (uint32_t i = 0; i < bufCount; i++) {
      bufInfos[i] = (VkDescriptorBufferInfo){
        .buffer = buffers[bufSlots[i]].buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
      };
      writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descSet,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufInfos[i],
      };
    }
    fp_vkUpdateDescriptorSets(device, bufCount, writes, 0, NULL);

    // Reset and re-record dispatch command buffer (no ONE_TIME_SUBMIT — reusable)
    fp_vkResetCommandBuffer(dispatchCmdBuf, 0);
    VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    fp_vkBeginCommandBuffer(dispatchCmdBuf, &beginInfo);
    fp_vkCmdBindPipeline(dispatchCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipeline);
    fp_vkCmdBindDescriptorSets(dispatchCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ps->layout, 0, 1, &descSet, 0, NULL);
    if (pushSize > 0) {
      fp_vkCmdPushConstants(dispatchCmdBuf, ps->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    }
    fp_vkCmdDispatch(dispatchCmdBuf, gX, gY, gZ);
    fp_vkEndCommandBuffer(dispatchCmdBuf);

    // Update cache
    cachedPipeSlot = pipeSlot;
    memcpy(cachedBufSlots, bufSlots, bufCount * sizeof(int32_t));
    cachedBufCount = bufCount;
    cachedGX = gX; cachedGY = gY; cachedGZ = gZ;
    cachedPushSize = pushSize;
    if (pushSize > 0) memcpy(cachedPushData, pushData, pushSize);
    dispatchCacheValid = 1;
  } else {
    // Cache hit: must wait for previous dispatch of this cmd buffer to finish
    // (can't have two submits of same non-simultaneous cmd buffer in flight)
    waitTimelineValue(lastDispatchTimeline);
  }

  // Submit async — signal timeline semaphore, don't wait
  uint64_t tv = submitCmdBufAsync(dispatchCmdBuf);
  if (tv == 0) {
    dispatchCacheValid = 0; // cmd buffer state is indeterminate after failed submit
    napi_throw_error(env, NULL, "dispatch: vkQueueSubmit failed");
    return NULL;
  }
  lastDispatchTimeline = tv;

  // Mark the last buffer (output) with the timeline value
  int32_t outSlot = bufSlots[bufCount - 1];
  if (outSlot >= 0 && outSlot < MAX_BUFFERS && buffers[outSlot].active) {
    buffers[outSlot].lastWriteTimeline = tv;
  }

  // Return timeline value to JS (as double since napi doesn't have uint64)
  napi_value result;
  napi_create_double(env, (double)tv, &result);
  return result;
}

// ── N-API: batchBegin() ─────────────────────────────────────────────────────

static napi_value napi_batchBegin(napi_env env, napi_callback_info info) {
  uint32_t slot = g_ringHead;
  // Only wait if THIS ring slot's GPU work hasn't completed yet
  if (g_ring[slot].timelineValue > 0) {
    uint64_t completed;
    fp_vkGetSemaphoreCounterValue(device, timelineSem, &completed);
    if (completed < g_ring[slot].timelineValue) {
      waitTimelineValue(g_ring[slot].timelineValue);
    }
  }

  if (g_ring[slot].descPool) {
    fp_vkResetDescriptorPool(device, g_ring[slot].descPool, 0);
  }
  dispatchCacheValid = 0; // batch uses the descriptor pool, invalidate single-dispatch cache

  fp_vkResetCommandBuffer(g_ring[slot].cmd, 0);
  VkCommandBufferBeginInfo beginInfo = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  fp_vkBeginCommandBuffer(g_ring[slot].cmd, &beginInfo);

  // Global memory barrier: ensure all shader writes from previously submitted
  // batches are visible to this batch. Without this, the ring's partial-wait
  // (only waiting on this slot, not the most recent batch) would allow stale reads.
  {
    VkMemoryBarrier memBarrier = {
      .sType = 46, // VK_STRUCTURE_TYPE_MEMORY_BARRIER
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    fp_vkCmdPipelineBarrier(g_ring[slot].cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &memBarrier, 0, NULL, 0, NULL);
  }

  batchRecording = 1;
  batchDispatchCount = 0;
  bufWriteGeneration++;
  return NULL;
}

// ── N-API: batchDispatch(pipeline, buffers[], gX, gY, gZ, pushConstants?, writeMask?) ──

static napi_value napi_batchDispatch(napi_env env, napi_callback_info info) {
  if (!batchRecording) {
    napi_throw_error(env, NULL, "batchDispatch called without batchBegin");
    return NULL;
  }

  size_t argc = 7;
  napi_value args[7];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t pipeSlot;
  napi_get_value_int32(env, args[0], &pipeSlot);

  if (pipeSlot < 0 || pipeSlot >= MAX_PIPELINES || !pipelines[pipeSlot].active) {
    napi_throw_error(env, NULL, "Invalid pipeline handle");
    return NULL;
  }
  PipelineSlot* ps = &pipelines[pipeSlot];

  uint32_t bufCount;
  napi_get_array_length(env, args[1], &bufCount);

  if (bufCount == 0 || bufCount > 32) {
    napi_throw_range_error(env, NULL, "batchDispatch: bufCount must be 1..32");
    return NULL;
  }

  int32_t bufSlots[32];
  for (uint32_t i = 0; i < bufCount; i++) {
    napi_value elem;
    napi_get_element(env, args[1], i, &elem);
    napi_get_value_int32(env, elem, &bufSlots[i]);
  }

  uint32_t gX = 1, gY = 1, gZ = 1;
  napi_get_value_uint32(env, args[2], &gX);
  if (argc > 3) napi_get_value_uint32(env, args[3], &gY);
  if (argc > 4) napi_get_value_uint32(env, args[4], &gZ);

  void* pushData = NULL;
  uint32_t pushSize = ps->pushConstantSize;
  if (argc > 5 && pushSize > 0) {
    napi_get_typedarray_info(env, args[5], NULL, NULL, &pushData, NULL, NULL);
  } else {
    pushSize = 0;
  }

  // Parse optional write mask (7th argument): bit i = buffer i is written
  uint32_t writeMask = 0;
  int hasWriteMask = 0;
  if (argc > 6) {
    void* wmData = NULL;
    size_t wmLen = 0;
    napi_status wmStatus = napi_get_typedarray_info(env, args[6], NULL, &wmLen, &wmData, NULL, NULL);
    if (wmStatus == napi_ok && wmData && wmLen > 0) {
      writeMask = ((uint32_t*)wmData)[0];
      hasWriteMask = 1;
    }
  }

  // Per-buffer barrier: only insert barriers for buffers that have RAW/WAR hazards
  if (batchDispatchCount > 0) {
    if (!hasWriteMask) {
      // Legacy fallback: global memory barrier (safe but conservative)
      VkMemoryBarrier barrier = {
        .sType = 46, // VK_STRUCTURE_TYPE_MEMORY_BARRIER
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      };
      fp_vkCmdPipelineBarrier(g_ring[g_ringHead].cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
    } else {
      // Fine-grained: emit VkBufferMemoryBarrier only for buffers with actual hazards
      // O(1) per buffer: check bufWriteGen[slot] == bufWriteGeneration
      VkBufferMemoryBarrier bufBarriers[32];
      uint32_t barrierCount = 0;

      for (uint32_t i = 0; i < bufCount; i++) {
        int32_t slot = bufSlots[i];
        if (slot >= 0 && slot < MAX_BUFFERS && bufWriteGen[slot] == bufWriteGeneration) {
          bufBarriers[barrierCount] = (VkBufferMemoryBarrier){
            .sType = 44, // VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = ((writeMask >> i) & 1)
              ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
              : VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = 0xFFFFFFFF, // VK_QUEUE_FAMILY_IGNORED
            .dstQueueFamilyIndex = 0xFFFFFFFF,
            .buffer = buffers[slot].buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
          };
          barrierCount++;
        }
      }

      if (barrierCount > 0) {
        fp_vkCmdPipelineBarrier(g_ring[g_ringHead].cmd,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0, 0, NULL, barrierCount, bufBarriers, 0, NULL);
      }
    }
  }

  // Record write tracking: O(1) per buffer via direct-indexed arrays
  if (hasWriteMask) {
    for (uint32_t i = 0; i < bufCount; i++) {
      if ((writeMask >> i) & 1) {
        bufWriteDispatch[bufSlots[i]] = batchDispatchCount;
        bufWriteGen[bufSlots[i]] = bufWriteGeneration;
      }
    }
  } else {
    // Without write mask, conservatively mark all buffers as written
    for (uint32_t i = 0; i < bufCount; i++) {
      bufWriteDispatch[bufSlots[i]] = batchDispatchCount;
      bufWriteGen[bufSlots[i]] = bufWriteGeneration;
    }
  }

  // Descriptors + dispatch into ring slot's command buffer
  VkDescriptorBufferInfo bufInfos[32];
  VkWriteDescriptorSet writes[32];
  VkCommandBuffer ringCmd = g_ring[g_ringHead].cmd;
  fp_vkCmdBindPipeline(ringCmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipeline);

  if (hasPushDescriptors && fp_vkCmdPushDescriptorSetKHR) {
    for (uint32_t i = 0; i < bufCount; i++) {
      bufInfos[i] = (VkDescriptorBufferInfo){
        .buffer = buffers[bufSlots[i]].buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
      };
      writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = 0,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufInfos[i],
      };
    }
    fp_vkCmdPushDescriptorSetKHR(ringCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      ps->layout, 0, bufCount, writes);
  } else {
    VkDescriptorSetAllocateInfo dsAllocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = g_ring[g_ringHead].descPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &ps->descLayout,
    };
    VkDescriptorSet descSet;
    VkResult res = fp_vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet);
    if (res != VK_SUCCESS) {
      napi_throw_error(env, NULL, "batch: vkAllocateDescriptorSets failed");
      return NULL;
    }
    for (uint32_t i = 0; i < bufCount; i++) {
      bufInfos[i] = (VkDescriptorBufferInfo){
        .buffer = buffers[bufSlots[i]].buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
      };
      writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descSet,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &bufInfos[i],
      };
    }
    fp_vkUpdateDescriptorSets(device, bufCount, writes, 0, NULL);
    fp_vkCmdBindDescriptorSets(ringCmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->layout, 0, 1, &descSet, 0, NULL);
  }
  if (pushSize > 0) {
    fp_vkCmdPushConstants(ringCmd, ps->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
  }
  fp_vkCmdDispatch(ringCmd, gX, gY, gZ);
  batchDispatchCount++;

  return NULL;
}

// ── N-API: batchSubmit() ────────────────────────────────────────────────────

static napi_value napi_batchSubmit(napi_env env, napi_callback_info info) {
  if (!batchRecording) {
    napi_throw_error(env, NULL, "batchSubmit called without batchBegin");
    return NULL;
  }

  uint32_t slot = g_ringHead;
  fp_vkEndCommandBuffer(g_ring[slot].cmd);

  uint64_t tv = 0;
  if (batchDispatchCount > 0) {
    tv = submitCmdBufAsync(g_ring[slot].cmd);
    if (tv == 0) {
      batchRecording = 0;
      batchDispatchCount = 0;
      napi_throw_error(env, NULL, "batchSubmit: vkQueueSubmit failed");
      return NULL;
    }
    g_ring[slot].timelineValue = tv;
    lastDispatchTimeline = tv;
  }

  // Advance ring head for the next batch
  g_ringHead = (g_ringHead + 1) % RING_SIZE;
  batchRecording = 0;
  batchDispatchCount = 0;

  napi_value result;
  napi_create_double(env, (double)tv, &result);
  return result;
}

// ── N-API: batchDispatchMany(packed: ArrayBuffer, count: number) ──────────────
//
// Packed binary format (little-endian), per dispatch, contiguous:
//   int32   pipelineSlot
//   uint16  bufCount
//   uint16  flags          // bits [15:1] = gY, bit 0 = hasGZ
//   uint32  gX
//   [uint32 gZ]            // only if hasGZ (bit 0 of flags)
//   uint32  writeMask
//   int32   bufHandles[bufCount]
//   uint8   pushData[pushSize]   // size from pipeline's pushConstantSize

static napi_value napi_batchDispatchMany(napi_env env, napi_callback_info info) {
  if (!batchRecording) {
    napi_throw_error(env, NULL, "batchDispatchMany called without batchBegin");
    return NULL;
  }

  size_t argc = 2;
  napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  void* packedData;
  size_t packedLen;
  napi_get_arraybuffer_info(env, args[0], &packedData, &packedLen);

  uint32_t count;
  napi_get_value_uint32(env, args[1], &count);

  VkCommandBuffer ringCmd = g_ring[g_ringHead].cmd;
  VkDescriptorPool ringDescPool = g_ring[g_ringHead].descPool;

  const uint8_t* ptr = (const uint8_t*)packedData;
  const uint8_t* end = ptr + packedLen;

  for (uint32_t d = 0; d < count; d++) {
    if (ptr + 12 > end) break; // minimum header: 4+2+2+4 = 12 bytes

    // Parse header
    int32_t pipeSlot;
    memcpy(&pipeSlot, ptr, 4); ptr += 4;

    uint16_t bufCount, flags;
    memcpy(&bufCount, ptr, 2); ptr += 2;
    memcpy(&flags, ptr, 2); ptr += 2;

    uint32_t gX;
    memcpy(&gX, ptr, 4); ptr += 4;

    uint32_t gY = (flags >> 1) & 0x7FFF;
    if (gY == 0) gY = 1;

    uint32_t gZ = 1;
    if (flags & 1) {
      if (ptr + 4 > end) break;
      memcpy(&gZ, ptr, 4); ptr += 4;
    }

    if (ptr + 4 > end) break;
    uint32_t writeMask;
    memcpy(&writeMask, ptr, 4); ptr += 4;

    // Parse buffer handles
    if (bufCount > 32 || ptr + bufCount * 4 > end) break;
    int32_t bufSlots[32];
    memcpy(bufSlots, ptr, bufCount * 4); ptr += bufCount * 4;

    // Validate pipeline
    if (pipeSlot < 0 || pipeSlot >= MAX_PIPELINES || !pipelines[pipeSlot].active) continue;
    PipelineSlot* ps = &pipelines[pipeSlot];

    // Parse push constants
    uint32_t pushSize = ps->pushConstantSize;
    const void* pushPtr = NULL;
    if (pushSize > 0) {
      if (ptr + pushSize > end) break;
      pushPtr = ptr;
      ptr += pushSize;
    }

    // ── Barriers: O(1) per buffer ──
    if (batchDispatchCount > 0) {
      VkBufferMemoryBarrier bufBarriers[32];
      uint32_t barrierCount = 0;

      for (uint32_t i = 0; i < bufCount; i++) {
        int32_t slot = bufSlots[i];
        if (slot >= 0 && slot < MAX_BUFFERS && bufWriteGen[slot] == bufWriteGeneration) {
          bufBarriers[barrierCount] = (VkBufferMemoryBarrier){
            .sType = 44, // VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = ((writeMask >> i) & 1)
              ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
              : VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = 0xFFFFFFFF,
            .dstQueueFamilyIndex = 0xFFFFFFFF,
            .buffer = buffers[slot].buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
          };
          barrierCount++;
        }
      }

      if (barrierCount > 0) {
        fp_vkCmdPipelineBarrier(ringCmd,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0, 0, NULL, barrierCount, bufBarriers, 0, NULL);
      }
    }

    // ── Write tracking: O(1) per buffer ──
    for (uint32_t i = 0; i < bufCount; i++) {
      if ((writeMask >> i) & 1) {
        bufWriteDispatch[bufSlots[i]] = batchDispatchCount;
        bufWriteGen[bufSlots[i]] = bufWriteGeneration;
      }
    }

    // ── Descriptors + dispatch ──
    VkDescriptorBufferInfo bufInfos[32];
    VkWriteDescriptorSet writes[32];

    fp_vkCmdBindPipeline(ringCmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipeline);

    if (hasPushDescriptors && fp_vkCmdPushDescriptorSetKHR) {
      // Push descriptors: write directly into command buffer, no pool allocation
      for (uint32_t i = 0; i < bufCount; i++) {
        bufInfos[i] = (VkDescriptorBufferInfo){
          .buffer = buffers[bufSlots[i]].buffer,
          .offset = 0,
          .range = VK_WHOLE_SIZE,
        };
        writes[i] = (VkWriteDescriptorSet){
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = 0, // ignored for push descriptors
          .dstBinding = i,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bufInfos[i],
        };
      }
      fp_vkCmdPushDescriptorSetKHR(ringCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        ps->layout, 0, bufCount, writes);
    } else {
      // Fallback: allocate from ring descriptor pool
      VkDescriptorSetAllocateInfo dsAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ringDescPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ps->descLayout,
      };
      VkDescriptorSet descSet;
      VkResult res = fp_vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet);
      if (res != VK_SUCCESS) {
        napi_throw_error(env, NULL, "batchDispatchMany: vkAllocateDescriptorSets failed");
        return NULL;
      }
      for (uint32_t i = 0; i < bufCount; i++) {
        bufInfos[i] = (VkDescriptorBufferInfo){
          .buffer = buffers[bufSlots[i]].buffer,
          .offset = 0,
          .range = VK_WHOLE_SIZE,
        };
        writes[i] = (VkWriteDescriptorSet){
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = descSet,
          .dstBinding = i,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bufInfos[i],
        };
      }
      fp_vkUpdateDescriptorSets(device, bufCount, writes, 0, NULL);
      fp_vkCmdBindDescriptorSets(ringCmd, VK_PIPELINE_BIND_POINT_COMPUTE, ps->layout, 0, 1, &descSet, 0, NULL);
    }

    if (pushSize > 0 && pushPtr) {
      fp_vkCmdPushConstants(ringCmd, ps->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushPtr);
    }
    fp_vkCmdDispatch(ringCmd, gX, gY, gZ);
    batchDispatchCount++;
  }

  return NULL;
}

// ── N-API: waitTimeline(value) ───────────────────────────────────────────────

static napi_value napi_waitTimeline(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  double val;
  napi_get_value_double(env, args[0], &val);
  waitTimelineValue((uint64_t)val);
  return NULL;
}

// ── N-API: getCompleted() → number ──────────────────────────────────────────

static napi_value napi_getCompleted(napi_env env, napi_callback_info info) {
  uint64_t completed = 0;
  if (timelineSem) {
    fp_vkGetSemaphoreCounterValue(device, timelineSem, &completed);
  }
  napi_value result;
  napi_create_double(env, (double)completed, &result);
  return result;
}

// ── N-API: gpuTime(pipeline, buffers[], gX, gY, gZ, pushConstants?) → microseconds ──

static napi_value napi_gpuTime(napi_env env, napi_callback_info info) {
  if (!timestampsSupported || !timestampPool) {
    napi_throw_error(env, NULL, "GPU timestamps not supported on this device");
    return NULL;
  }

  size_t argc = 6;
  napi_value args[6];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t pipeSlot;
  napi_get_value_int32(env, args[0], &pipeSlot);

  if (pipeSlot < 0 || pipeSlot >= MAX_PIPELINES || !pipelines[pipeSlot].active) {
    napi_throw_error(env, NULL, "Invalid pipeline handle");
    return NULL;
  }
  PipelineSlot* ps = &pipelines[pipeSlot];

  uint32_t bufCount;
  napi_get_array_length(env, args[1], &bufCount);

  if (bufCount == 0 || bufCount > 32) {
    napi_throw_range_error(env, NULL, "gpuTime: bufCount must be 1..32");
    return NULL;
  }

  int32_t bufSlots[32];
  for (uint32_t i = 0; i < bufCount; i++) {
    napi_value elem;
    napi_get_element(env, args[1], i, &elem);
    napi_get_value_int32(env, elem, &bufSlots[i]);
  }

  uint32_t gX = 1, gY = 1, gZ = 1;
  napi_get_value_uint32(env, args[2], &gX);
  if (argc > 3) napi_get_value_uint32(env, args[3], &gY);
  if (argc > 4) napi_get_value_uint32(env, args[4], &gZ);

  void* pushDataPtr = NULL;
  uint32_t pushSize = ps->pushConstantSize;
  if (argc > 5 && pushSize > 0) {
    napi_get_typedarray_info(env, args[5], NULL, NULL, &pushDataPtr, NULL, NULL);
  } else {
    pushSize = 0;
  }

  // Wait for any in-flight work
  waitTimelineValue(lastDispatchTimeline);

  // Reset descriptor pool and allocate set
  fp_vkResetDescriptorPool(device, singleDescPool, 0);
  dispatchCacheValid = 0;

  VkDescriptorSetAllocateInfo dsAllocInfo = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = singleDescPool,
    .descriptorSetCount = 1,
    .pSetLayouts = &ps->descLayout,
  };
  VkDescriptorSet descSet;
  fp_vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet);

  VkDescriptorBufferInfo bufInfos[32];
  VkWriteDescriptorSet writes[32];
  for (uint32_t i = 0; i < bufCount; i++) {
    bufInfos[i] = (VkDescriptorBufferInfo){
      .buffer = buffers[bufSlots[i]].buffer,
      .offset = 0,
      .range = VK_WHOLE_SIZE,
    };
    writes[i] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descSet,
      .dstBinding = i,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bufInfos[i],
    };
  }
  fp_vkUpdateDescriptorSets(device, bufCount, writes, 0, NULL);

  // Record command buffer with timestamps
  fp_vkResetCommandBuffer(dispatchCmdBuf, 0);
  VkCommandBufferBeginInfo beginInfo = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  fp_vkBeginCommandBuffer(dispatchCmdBuf, &beginInfo);

  // Reset queries and write start timestamp
  fp_vkCmdResetQueryPool(dispatchCmdBuf, timestampPool, 0, 2);
  fp_vkCmdWriteTimestamp(dispatchCmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool, 0);

  fp_vkCmdBindPipeline(dispatchCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipeline);
  fp_vkCmdBindDescriptorSets(dispatchCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ps->layout, 0, 1, &descSet, 0, NULL);
  if (pushSize > 0) {
    fp_vkCmdPushConstants(dispatchCmdBuf, ps->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushDataPtr);
  }
  fp_vkCmdDispatch(dispatchCmdBuf, gX, gY, gZ);

  // Write end timestamp
  fp_vkCmdWriteTimestamp(dispatchCmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool, 1);
  fp_vkEndCommandBuffer(dispatchCmdBuf);

  // Submit synchronously with fence
  submitCmdBufSync(dispatchCmdBuf);

  // Read timestamps
  uint64_t timestamps[2] = {0, 0};
  fp_vkGetQueryPoolResults(device, timestampPool, 0, 2,
    sizeof(timestamps), timestamps, sizeof(uint64_t),
    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

  double elapsedUs = (double)(timestamps[1] - timestamps[0]) * (double)timestampPeriodNs / 1000.0;

  napi_value result;
  napi_create_double(env, elapsedUs, &result);
  return result;
}

// ── N-API: waitIdle() ───────────────────────────────────────────────────────

static napi_value napi_waitIdle(napi_env env, napi_callback_info info) {
  if (device) fp_vkDeviceWaitIdle(device);
  return NULL;
}

// ── N-API: destroy() ────────────────────────────────────────────────────────

static napi_value napi_destroy(napi_env env, napi_callback_info info) {
  if (!device) return NULL;

  fp_vkDeviceWaitIdle(device);

  // Destroy buffers (VkBuffer handles only — slab memory freed below)
  for (int i = 0; i < MAX_BUFFERS; i++) {
    if (buffers[i].active) {
      fp_vkDestroyBuffer(device, buffers[i].buffer, NULL);
      if (!buffers[i].slabAllocated && buffers[i].memory) {
        if (buffers[i].mapped) fp_vkUnmapMemory(device, buffers[i].memory);
        fp_vkFreeMemory(device, buffers[i].memory, NULL);
      }
      buffers[i].active = 0;
    }
  }

  // Destroy slab pools
  for (uint32_t i = 0; i < devicePool.slabCount; i++) {
    if (devicePool.slabs[i].mapped) fp_vkUnmapMemory(device, devicePool.slabs[i].memory);
    fp_vkFreeMemory(device, devicePool.slabs[i].memory, NULL);
  }
  memset(&devicePool, 0, sizeof(devicePool));
  for (uint32_t i = 0; i < hostPool.slabCount; i++) {
    if (hostPool.slabs[i].mapped) fp_vkUnmapMemory(device, hostPool.slabs[i].memory);
    fp_vkFreeMemory(device, hostPool.slabs[i].memory, NULL);
  }
  memset(&hostPool, 0, sizeof(hostPool));

  // Destroy pipelines
  for (int i = 0; i < MAX_PIPELINES; i++) {
    if (pipelines[i].active) {
      fp_vkDestroyPipeline(device, pipelines[i].pipeline, NULL);
      fp_vkDestroyPipelineLayout(device, pipelines[i].layout, NULL);
      fp_vkDestroyDescriptorSetLayout(device, pipelines[i].descLayout, NULL);
      pipelines[i].active = 0;
    }
  }

  // Destroy staging ring
  for (int si = 0; si < STAGING_RING_SIZE; si++) {
    if (stagingRing[si].buffer) {
      fp_vkUnmapMemory(device, stagingRing[si].memory);
      fp_vkDestroyBuffer(device, stagingRing[si].buffer, NULL);
      fp_vkFreeMemory(device, stagingRing[si].memory, NULL);
      memset(&stagingRing[si], 0, sizeof(StagingSlot));
    }
  }
  stagingRingInited = 0;

  // Destroy legacy staging buffer
  if (stagingBuffer != 0) {
    fp_vkUnmapMemory(device, stagingMemory);
    fp_vkDestroyBuffer(device, stagingBuffer, NULL);
    fp_vkFreeMemory(device, stagingMemory, NULL);
    stagingBuffer = 0; stagingMemory = 0; stagingMapped = NULL; stagingSize = 0;
  }

  // Destroy transfer queue resources
  if (transferFence) { fp_vkDestroyFence(device, transferFence, NULL); transferFence = 0; }
  if (transferCmdPool) { fp_vkDestroyCommandPool(device, transferCmdPool, NULL); transferCmdPool = NULL; }
  xferCmdBuf = NULL;
  transferQueue = NULL;
  transferQueueFamily = UINT32_MAX;
  hasAsyncTransfer = 0;

  if (timestampPool) { fp_vkDestroyQueryPool(device, timestampPool, NULL); timestampPool = 0; }
  timestampsSupported = 0;
  fp_vkDestroyFence(device, persistentFence, NULL);
  persistentFence = 0;
  if (timelineSem) { fp_vkDestroySemaphore(device, timelineSem, NULL); timelineSem = 0; }
  nextTimelineValue = 1;
  lastDispatchTimeline = 0;

  // Destroy ring resources
  for (int ri = 0; ri < RING_SIZE; ri++) {
    if (g_ring[ri].descPool) { fp_vkDestroyDescriptorPool(device, g_ring[ri].descPool, NULL); g_ring[ri].descPool = 0; }
    if (g_ring[ri].cmdPool) { fp_vkDestroyCommandPool(device, g_ring[ri].cmdPool, NULL); g_ring[ri].cmdPool = NULL; }
    g_ring[ri].cmd = NULL;
    g_ring[ri].timelineValue = 0;
  }
  g_ringHead = 0;

  if (singleDescPool) { fp_vkDestroyDescriptorPool(device, singleDescPool, NULL); singleDescPool = 0; }
  dispatchCmdBuf = NULL;
  transferCmdBuf = NULL;
  batchRecording = 0;
  batchDispatchCount = 0;
  dispatchCacheValid = 0;
  cachedPushSize = 0;
  fp_vkDestroyCommandPool(device, cmdPool, NULL);
  fp_vkDestroyDevice(device, NULL);
  fp_vkDestroyInstance(instance, NULL);

  device = NULL;
  instance = NULL;
  if (vk_lib) { dlclose(vk_lib); vk_lib = NULL; }

  return NULL;
}

// ── Module init ─────────────────────────────────────────────────────────────

static napi_value Init(napi_env env, napi_value exports) {
  napi_property_descriptor props[] = {
    { "initDevice",      NULL, napi_initDevice,      NULL, NULL, NULL, napi_default, NULL },
    { "createBuffer",    NULL, napi_createBuffer,    NULL, NULL, NULL, napi_default, NULL },
    { "uploadBuffer",    NULL, napi_uploadBuffer,    NULL, NULL, NULL, napi_default, NULL },
    { "readBuffer",      NULL, napi_readBuffer,      NULL, NULL, NULL, napi_default, NULL },
    { "destroyBuffer",   NULL, napi_destroyBuffer,   NULL, NULL, NULL, napi_default, NULL },
    { "createPipeline",  NULL, napi_createPipeline,  NULL, NULL, NULL, napi_default, NULL },
    { "dispatch",        NULL, napi_dispatch,        NULL, NULL, NULL, napi_default, NULL },
    { "batchBegin",      NULL, napi_batchBegin,      NULL, NULL, NULL, napi_default, NULL },
    { "batchDispatch",   NULL, napi_batchDispatch,   NULL, NULL, NULL, napi_default, NULL },
    { "batchDispatchMany", NULL, napi_batchDispatchMany, NULL, NULL, NULL, napi_default, NULL },
    { "batchSubmit",     NULL, napi_batchSubmit,     NULL, NULL, NULL, napi_default, NULL },
    { "waitTimeline",    NULL, napi_waitTimeline,    NULL, NULL, NULL, napi_default, NULL },
    { "getCompleted",    NULL, napi_getCompleted,    NULL, NULL, NULL, napi_default, NULL },
    { "gpuTime",         NULL, napi_gpuTime,         NULL, NULL, NULL, napi_default, NULL },
    { "waitIdle",        NULL, napi_waitIdle,        NULL, NULL, NULL, napi_default, NULL },
    { "destroy",         NULL, napi_destroy,         NULL, NULL, NULL, napi_default, NULL },
  };
  napi_define_properties(env, exports, sizeof(props) / sizeof(props[0]), props);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)

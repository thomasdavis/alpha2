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
  VK_QUEUE_COMPUTE_BIT = 0x00000002,
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
typedef VkResult (*PFN_vkFlushMappedMemoryRanges)(VkDevice, uint32_t, const VkMappedMemoryRange*);
typedef VkResult (*PFN_vkInvalidateMappedMemoryRanges)(VkDevice, uint32_t, const VkMappedMemoryRange*);
typedef void     (*PFN_vkCmdCopyBuffer)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const void*);

// Buffer copy region
typedef struct { VkDeviceSize srcOffset; VkDeviceSize dstOffset; VkDeviceSize size; } VkBufferCopy;

// ── Global state ────────────────────────────────────────────────────────────

static void* vk_lib = NULL;
static VkInstance instance = NULL;
static VkPhysicalDevice physDevice = NULL;
static VkDevice device = NULL;
static VkQueue computeQueue = NULL;
static VkCommandPool cmdPool = NULL;
static uint32_t computeQueueFamily = 0;
static VkPhysicalDeviceMemoryProperties memProps;
static char deviceNameStr[256] = {0};
static uint32_t vendorId = 0;

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
static PFN_vkFlushMappedMemoryRanges              fp_vkFlushMappedMemoryRanges;
static PFN_vkInvalidateMappedMemoryRanges         fp_vkInvalidateMappedMemoryRanges;
static PFN_vkCmdCopyBuffer                        fp_vkCmdCopyBuffer;

// ── Resource tracking ───────────────────────────────────────────────────────

#define MAX_BUFFERS   1024
#define MAX_PIPELINES 128

typedef struct {
  VkBuffer       buffer;
  VkDeviceMemory memory;
  VkDeviceSize   size;
  int            hostVisible; // 1 if host-visible
  int            active;
} BufferSlot;

typedef struct {
  VkPipeline            pipeline;
  VkPipelineLayout      layout;
  VkDescriptorSetLayout descLayout;
  uint32_t              numBindings;
  int                   active;
} PipelineSlot;

static BufferSlot   buffers[MAX_BUFFERS];
static PipelineSlot pipelines[MAX_PIPELINES];

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
  LOAD_VK(vkFlushMappedMemoryRanges);
  LOAD_VK(vkInvalidateMappedMemoryRanges);
  LOAD_VK(vkCmdCopyBuffer);

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
  for (uint32_t i = 0; i < qfCount; i++) {
    if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      computeQueueFamily = i;
      break;
    }
  }
  free(qfProps);

  if (computeQueueFamily == UINT32_MAX) {
    napi_throw_error(env, NULL, "No compute queue family found");
    return NULL;
  }

  // Create logical device
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queueCreate = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = computeQueueFamily,
    .queueCount = 1,
    .pQueuePriorities = &priority,
  };
  VkDeviceCreateInfo devCreate = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &queueCreate,
  };
  res = fp_vkCreateDevice(physDevice, &devCreate, NULL, &device);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateDevice failed");
    return NULL;
  }

  fp_vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

  // Create command pool
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

  // Init resource slots
  memset(buffers, 0, sizeof(buffers));
  memset(pipelines, 0, sizeof(pipelines));

  // Return device info
  napi_value result, val;
  napi_create_object(env, &result);

  napi_create_string_utf8(env, deviceNameStr, strlen(deviceNameStr), &val);
  napi_set_named_property(env, result, "deviceName", val);

  napi_create_uint32(env, vendorId, &val);
  napi_set_named_property(env, result, "vendorId", val);

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

  VkBufferCreateInfo bufInfo = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = byteLength,
    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VkResult res = fp_vkCreateBuffer(device, &bufInfo, NULL, &buffers[slot].buffer);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateBuffer failed");
    return NULL;
  }

  VkMemoryRequirements memReq;
  fp_vkGetBufferMemoryRequirements(device, buffers[slot].buffer, &memReq);

  VkFlags memFlags = hostVisible
    ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  uint32_t memType = findMemoryType(memReq.memoryTypeBits, memFlags);

  // Fall back to host-visible if device-local not available (integrated GPU)
  if (memType == UINT32_MAX && !hostVisible) {
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
  res = fp_vkAllocateMemory(device, &allocInfo, NULL, &buffers[slot].memory);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkAllocateMemory failed");
    return NULL;
  }

  fp_vkBindBufferMemory(device, buffers[slot].buffer, buffers[slot].memory, 0);
  buffers[slot].size = byteLength;
  buffers[slot].hostVisible = hostVisible;
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

  void* mapped;
  VkResult res = fp_vkMapMemory(device, buffers[slot].memory, 0, buffers[slot].size, 0, &mapped);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkMapMemory failed");
    return NULL;
  }

  size_t copyLen = byteLength < buffers[slot].size ? byteLength : buffers[slot].size;
  memcpy(mapped, data, copyLen);
  fp_vkUnmapMemory(device, buffers[slot].memory);

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

  void* mapped;
  VkResult res = fp_vkMapMemory(device, buffers[slot].memory, 0, buffers[slot].size, 0, &mapped);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkMapMemory failed for read");
    return NULL;
  }

  uint32_t elemCount = (uint32_t)(buffers[slot].size / 4);
  napi_value arrayBuf, result;
  void* outData;
  napi_create_arraybuffer(env, buffers[slot].size, &outData, &arrayBuf);
  memcpy(outData, mapped, buffers[slot].size);
  fp_vkUnmapMemory(device, buffers[slot].memory);

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
    fp_vkFreeMemory(device, buffers[slot].memory, NULL);
    buffers[slot].active = 0;
  }

  return NULL;
}

// ── N-API: createPipeline(spirvUint32Array, numBindings) → handle ───────────

static napi_value napi_createPipeline(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  void* spirvData;
  size_t spirvLen; // element count (uint32)
  napi_get_typedarray_info(env, args[0], NULL, &spirvLen, &spirvData, NULL, NULL);

  uint32_t numBindings = 4;
  if (argc > 1) napi_get_value_uint32(env, args[1], &numBindings);

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

  // Create pipeline layout
  VkPipelineLayoutCreateInfo layoutInfo = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .setLayoutCount = 1,
    .pSetLayouts = &pipelines[slot].descLayout,
  };
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
  pipelines[slot].active = 1;

  napi_value result;
  napi_create_int32(env, slot, &result);
  return result;
}

// ── N-API: dispatch(pipelineHandle, bufferHandles[], gX, gY, gZ) ────────────

static napi_value napi_dispatch(napi_env env, napi_callback_info info) {
  size_t argc = 5;
  napi_value args[5];
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

  int32_t bufSlots[32];
  for (uint32_t i = 0; i < bufCount && i < 32; i++) {
    napi_value elem;
    napi_get_element(env, args[1], i, &elem);
    napi_get_value_int32(env, elem, &bufSlots[i]);
  }

  uint32_t gX = 1, gY = 1, gZ = 1;
  napi_get_value_uint32(env, args[2], &gX);
  if (argc > 3) napi_get_value_uint32(env, args[3], &gY);
  if (argc > 4) napi_get_value_uint32(env, args[4], &gZ);

  PipelineSlot* ps = &pipelines[pipeSlot];

  // Create descriptor pool
  VkDescriptorPoolSize poolSize = {
    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = ps->numBindings,
  };
  VkDescriptorPoolCreateInfo dpInfo = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .maxSets = 1,
    .poolSizeCount = 1,
    .pPoolSizes = &poolSize,
  };
  VkDescriptorPool descPool;
  VkResult res = fp_vkCreateDescriptorPool(device, &dpInfo, NULL, &descPool);
  if (res != VK_SUCCESS) {
    napi_throw_error(env, NULL, "vkCreateDescriptorPool failed");
    return NULL;
  }

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo dsAllocInfo = {
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    .descriptorPool = descPool,
    .descriptorSetCount = 1,
    .pSetLayouts = &ps->descLayout,
  };
  VkDescriptorSet descSet;
  res = fp_vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet);
  if (res != VK_SUCCESS) {
    fp_vkDestroyDescriptorPool(device, descPool, NULL);
    napi_throw_error(env, NULL, "vkAllocateDescriptorSets failed");
    return NULL;
  }

  // Write descriptors
  VkDescriptorBufferInfo* bufInfos = malloc(sizeof(VkDescriptorBufferInfo) * bufCount);
  VkWriteDescriptorSet* writes = malloc(sizeof(VkWriteDescriptorSet) * bufCount);
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
  free(bufInfos);
  free(writes);

  // Allocate command buffer
  VkCommandBufferAllocateInfo cbAllocInfo = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = cmdPool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = 1,
  };
  VkCommandBuffer cmdBuf;
  fp_vkAllocateCommandBuffers(device, &cbAllocInfo, &cmdBuf);

  // Record commands
  VkCommandBufferBeginInfo beginInfo = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  fp_vkBeginCommandBuffer(cmdBuf, &beginInfo);
  fp_vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ps->pipeline);
  fp_vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ps->layout, 0, 1, &descSet, 0, NULL);
  fp_vkCmdDispatch(cmdBuf, gX, gY, gZ);
  fp_vkEndCommandBuffer(cmdBuf);

  // Submit and wait
  VkSubmitInfo submitInfo = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &cmdBuf,
  };
  fp_vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
  fp_vkQueueWaitIdle(computeQueue);

  // Cleanup
  fp_vkResetCommandPool(device, cmdPool, 0);
  fp_vkDestroyDescriptorPool(device, descPool, NULL);

  return NULL;
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

  // Destroy buffers
  for (int i = 0; i < MAX_BUFFERS; i++) {
    if (buffers[i].active) {
      fp_vkDestroyBuffer(device, buffers[i].buffer, NULL);
      fp_vkFreeMemory(device, buffers[i].memory, NULL);
      buffers[i].active = 0;
    }
  }

  // Destroy pipelines
  for (int i = 0; i < MAX_PIPELINES; i++) {
    if (pipelines[i].active) {
      fp_vkDestroyPipeline(device, pipelines[i].pipeline, NULL);
      fp_vkDestroyPipelineLayout(device, pipelines[i].layout, NULL);
      fp_vkDestroyDescriptorSetLayout(device, pipelines[i].descLayout, NULL);
      pipelines[i].active = 0;
    }
  }

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
    { "initDevice",     NULL, napi_initDevice,     NULL, NULL, NULL, napi_default, NULL },
    { "createBuffer",   NULL, napi_createBuffer,   NULL, NULL, NULL, napi_default, NULL },
    { "uploadBuffer",   NULL, napi_uploadBuffer,   NULL, NULL, NULL, napi_default, NULL },
    { "readBuffer",     NULL, napi_readBuffer,     NULL, NULL, NULL, napi_default, NULL },
    { "destroyBuffer",  NULL, napi_destroyBuffer,  NULL, NULL, NULL, napi_default, NULL },
    { "createPipeline", NULL, napi_createPipeline, NULL, NULL, NULL, napi_default, NULL },
    { "dispatch",       NULL, napi_dispatch,       NULL, NULL, NULL, napi_default, NULL },
    { "waitIdle",       NULL, napi_waitIdle,       NULL, NULL, NULL, napi_default, NULL },
    { "destroy",        NULL, napi_destroy,        NULL, NULL, NULL, napi_default, NULL },
  };
  napi_define_properties(env, exports, sizeof(props) / sizeof(props[0]), props);
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)

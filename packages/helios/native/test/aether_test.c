/*
 * aether_test.c — known-answer tests for the ioctl transport.
 *
 * WHAT: struct layout checks against the vendor ABI, and ioctl request-code
 * assembly checked against hand-computed values.
 *
 * WHY: nothing here needs a GPU, which is the point — every one of these
 * failures would otherwise show up on rented hardware as an ioctl returning
 * EINVAL, or worse, as RM reading a field we never set. Catching them on a box
 * with no GPU at all is free.
 *
 * The expected values are computed by hand from the C ABI rules and from the
 * Linux _IOC bit layout, not by printing what the code produces and pasting it
 * back. A test whose expectation came from the implementation proves only that
 * the implementation is deterministic (standard 5).
 */
#include "../aether/device.h"
#include "../aether/ioctl.h"
#include "../aether/nv_abi.h"
#include "harness.h"

static void test_scalar_widths(void) {
  HT_CASE("vendor scalar types are the widths RM assumes");
  HT_SIZEOF(NvHandle, 4);
  HT_SIZEOF(NvU32, 4);
  HT_SIZEOF(NvV32, 4);
  HT_SIZEOF(NvU64, 8);
  /* NvP64 is 64-bit even on a 32-bit host — that is the whole reason RM does
   * not just use a pointer here. */
  HT_SIZEOF(NvP64, 8);
  HT_END();
}

static void test_nvos21_layout(void) {
  /* Four 32-bit fields, then an 8-aligned 64-bit pointer, then two 32-bit.
   * Offsets follow from that directly:
   *   hRoot 0, hObjectParent 4, hObjectNew 8, hClass 12,
   *   pAllocParms 16 (already aligned), paramsSize 24, status 28, size 32. */
  HT_CASE("NVOS21_PARAMETERS layout (alloc object)");
  HT_OFFSET(NVOS21_PARAMETERS, hRoot, 0);
  HT_OFFSET(NVOS21_PARAMETERS, hObjectParent, 4);
  HT_OFFSET(NVOS21_PARAMETERS, hObjectNew, 8);
  HT_OFFSET(NVOS21_PARAMETERS, hClass, 12);
  HT_OFFSET(NVOS21_PARAMETERS, pAllocParms, 16);
  HT_OFFSET(NVOS21_PARAMETERS, paramsSize, 24);
  HT_OFFSET(NVOS21_PARAMETERS, status, 28);
  HT_SIZEOF(NVOS21_PARAMETERS, 32);
  HT_END();
}

static void test_nvos54_layout(void) {
  /* hClient 0, hObject 4, cmd 8, flags 12, params 16, paramsSize 24,
   * status 28, size 32. */
  HT_CASE("NVOS54_PARAMETERS layout (control)");
  HT_OFFSET(NVOS54_PARAMETERS, hClient, 0);
  HT_OFFSET(NVOS54_PARAMETERS, hObject, 4);
  HT_OFFSET(NVOS54_PARAMETERS, cmd, 8);
  HT_OFFSET(NVOS54_PARAMETERS, flags, 12);
  HT_OFFSET(NVOS54_PARAMETERS, params, 16);
  HT_OFFSET(NVOS54_PARAMETERS, paramsSize, 24);
  HT_OFFSET(NVOS54_PARAMETERS, status, 28);
  HT_SIZEOF(NVOS54_PARAMETERS, 32);
  HT_END();
}

static void test_nvos33_layout(void) {
  /* Three 32-bit handles then an 8-aligned u64 — so there IS four bytes of
   * padding after hMemory. Getting this wrong shifts every subsequent field
   * and is precisely the failure this file exists to catch.
   *   hClient 0, hDevice 4, hMemory 8, <pad 12>, offset 16, length 24,
   *   pLinearAddress 32, status 40, flags 44, size 48. */
  HT_CASE("NVOS33_PARAMETERS layout (map memory, has padding)");
  HT_OFFSET(NVOS33_PARAMETERS, hClient, 0);
  HT_OFFSET(NVOS33_PARAMETERS, hDevice, 4);
  HT_OFFSET(NVOS33_PARAMETERS, hMemory, 8);
  HT_OFFSET(NVOS33_PARAMETERS, offset, 16);
  HT_OFFSET(NVOS33_PARAMETERS, length, 24);
  HT_OFFSET(NVOS33_PARAMETERS, pLinearAddress, 32);
  HT_OFFSET(NVOS33_PARAMETERS, status, 40);
  HT_OFFSET(NVOS33_PARAMETERS, flags, 44);
  HT_SIZEOF(NVOS33_PARAMETERS, 48);
  HT_END();
}

static void test_nvos02_layout(void) {
  /* hRoot 0, hObjectParent 4, hObjectNew 8, hClass 12, flags 16,
   * <pad 20>, pMemory 24, limit 32, status 40, tail padding to 48. */
  HT_CASE("NVOS02_PARAMETERS layout (alloc memory)");
  HT_OFFSET(NVOS02_PARAMETERS, hRoot, 0);
  HT_OFFSET(NVOS02_PARAMETERS, hObjectParent, 4);
  HT_OFFSET(NVOS02_PARAMETERS, hObjectNew, 8);
  HT_OFFSET(NVOS02_PARAMETERS, hClass, 12);
  HT_OFFSET(NVOS02_PARAMETERS, flags, 16);
  HT_OFFSET(NVOS02_PARAMETERS, pMemory, 24);
  HT_OFFSET(NVOS02_PARAMETERS, limit, 32);
  HT_OFFSET(NVOS02_PARAMETERS, status, 40);
  HT_END();
}

static void test_ioctl_request_encoding(void) {
  /* Hand-computed from the _IOC layout in ioctl.h:
   *   dir  = READ|WRITE = 3, at bit 30
   *   type = 'F' = 0x46, at bit 8
   *   nr   = escape code, at bit 0
   *   size = sizeof(params), at bit 16
   *
   * For NV_ESC_RM_ALLOC (0x2B) with a 32-byte NVOS21:
   *   (3 << 30) | (32 << 16) | (0x46 << 8) | 0x2B
   *   = 0xC0000000 | 0x00200000 | 0x00004600 | 0x2B
   *   = 0xC020462B
   */
  HT_CASE("ioctl request codes match hand-computed values");
  HT_EQ_U64(AE_IOWR(NV_ESC_RM_ALLOC, sizeof(NVOS21_PARAMETERS)), 0xC020462BUL);

  /* NV_ESC_RM_CONTROL (0x2A), 32-byte NVOS54 -> same size field, nr 0x2A. */
  HT_EQ_U64(AE_IOWR(NV_ESC_RM_CONTROL, sizeof(NVOS54_PARAMETERS)), 0xC020462AUL);

  /* NV_ESC_RM_MAP_MEMORY (0x4E), 48-byte NVOS33:
   *   (3 << 30) | (48 << 16) | (0x46 << 8) | 0x4E = 0xC030464E */
  HT_EQ_U64(AE_IOWR(NV_ESC_RM_MAP_MEMORY, sizeof(NVOS33_PARAMETERS)),
            0xC030464EUL);

  /* The size field is 14 bits; a struct at the boundary must not overflow into
   * the direction bits. 16383 is the largest representable size. */
  HT_EQ_U64((AE_IOWR(0, 16383) >> AE_IOC_DIRSHIFT) & 0x3, 3);
  HT_END();
}

static void test_class_ids(void) {
  /* These are the identifiers RM matches against when allocating an object;
   * a wrong one produces NV_ERR_INVALID_CLASS at runtime on hardware only. */
  HT_CASE("RM class ids match the vendor headers");
  HT_EQ_U64(NV01_ROOT_CLIENT, 0x41);
  HT_EQ_U64(NV01_DEVICE_0, 0x80);
  HT_EQ_U64(NV20_SUBDEVICE_0, 0x2080);
  HT_EQ_U64(FERMI_VASPACE_A, 0x90f1);
  HT_EQ_U64(AMPERE_CHANNEL_GPFIFO_A, 0xc56f);
  /* sm_86 is GA10x, which is AMPERE_COMPUTE_B. AMPERE_COMPUTE_A (0xc6c0) is
   * GA100 and is NOT what a 3070/3080/3090 exposes. */
  HT_EQ_U64(AMPERE_COMPUTE_B, 0xc7c0);
  HT_END();
}

static void test_status_names(void) {
  HT_CASE("status decoding keeps unknown codes greppable");
  HT_TRUE(strcmp(aether_status_name(NV_OK), "NV_OK") == 0);
  HT_TRUE(strcmp(aether_status_name(NV_ERR_NOT_SUPPORTED),
                 "NV_ERR_NOT_SUPPORTED") == 0);
  /* An unrecognised status must still carry its numeric value, so it can be
   * looked up in nvstatuscodes.h rather than vanishing into "unknown". */
  HT_TRUE(strstr(aether_status_name(0x12345678), "12345678") != NULL);
  HT_END();
}

/* The handle allocator is ours, not RM's — a collision aliases two objects, so
 * it is worth proving it is monotonic and starts where we think. */
static void test_handle_allocator(void) {
  HT_CASE("handle allocator is monotonic and namespaced");
  aether_device d;
  memset(&d, 0, sizeof d);
  d.nextHandle = 0xcafe0000u;

  NvHandle a = aether_next_handle(&d);
  NvHandle b = aether_next_handle(&d);
  NvHandle c = aether_next_handle(&d);
  HT_EQ_U64(a, 0xcafe0000u);
  HT_EQ_U64(b, 0xcafe0001u);
  HT_EQ_U64(c, 0xcafe0002u);
  HT_TRUE(a != b && b != c);
  HT_END();
}

/* Opening a device on a box with no GPU must fail cleanly rather than crash or
 * leave half a chain behind. This runs everywhere, including CI. */
static void test_open_without_gpu_is_clean(void) {
  HT_CASE("device open fails cleanly when there is no GPU");
  aether_device d;
  int rc = aether_device_open(&d, 0);
  if (rc == 0) {
    /* A real GPU is present — then the chain must be fully built. */
    HT_TRUE(d.client != 0);
    HT_TRUE(d.device != 0);
    HT_TRUE(d.subdevice != 0);
    HT_TRUE(d.vaspace != 0);
    aether_device_close(&d);
    HT_EQ_U64(d.client, 0);
  } else {
    /* No GPU: everything must be reset, not left dangling. */
    HT_EQ_U64(d.client, 0);
    HT_EQ_U64(d.device, 0);
    HT_TRUE(d.ctlFd < 0);
    HT_TRUE(d.gpuFd < 0);
  }
  HT_END();
}

void ht_run(void) {
  printf("\naether — ioctl transport\n");
  test_scalar_widths();
  test_nvos21_layout();
  test_nvos54_layout();
  test_nvos33_layout();
  test_nvos02_layout();
  test_ioctl_request_encoding();
  test_class_ids();
  test_status_names();
  test_handle_allocator();
  test_open_without_gpu_is_clean();
}

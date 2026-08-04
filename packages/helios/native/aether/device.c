/*
 * device.c — opening a GPU and building its object chain.
 *
 * WHAT: enumerate the cards this process can actually reach, open one, and
 * build the client/device/subdevice/address-space chain everything else hangs
 * off. The RM verbs those calls are made with live in rm.c.
 */
/* O_CLOEXEC is POSIX-2008, not C11, so the build uses -std=gnu11. A
 * _GNU_SOURCE define here would be a no-op: it has to precede the first system
 * header, and device.h pulls one in. Putting the requirement in the build flags
 * rather than in a define that silently does nothing. */
#include "device.h"
#include "ioctl.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Allocation parameters, transcribed from the vendor SDK. Layout is asserted in
 * aether_test.c for the same reason as the NVOS structs: a wrong offset here is
 * a kernel reading the wrong field, not a compile error.
 *
 * PROVENANCE: sdk/nvidia/inc/class/cl0080.h, cl2080.h, and nvos.h. */
typedef struct {
  NvU32 deviceId;
  NvHandle hClientShare;
  NvHandle hTargetClient;
  NvHandle hTargetDevice;
  NvV32 flags;
  NvU64 vaSpaceSize __attribute__((aligned(8)));
  NvU64 vaStartInternal __attribute__((aligned(8)));
  NvU64 vaLimitInternal __attribute__((aligned(8)));
  NvV32 vaMode;
} NV0080_ALLOC_PARAMETERS;

typedef struct {
  NvU32 subDeviceId;
} NV2080_ALLOC_PARAMETERS;

typedef struct {
  NvU32 index;
  NvV32 flags;
  NvU64 vaSize __attribute__((aligned(8)));
  NvU64 vaStartInternal __attribute__((aligned(8)));
  NvU64 vaLimitInternal __attribute__((aligned(8)));
  NvU32 bigPageSize;
  NvU64 vaBase __attribute__((aligned(8)));
  NvU32 pasid;
} NV_VASPACE_ALLOCATION_PARAMETERS;

/* Handles are ours to choose. Starting well above zero keeps them visually
 * distinct from the small constants RM itself uses in traces. */
#define AETHER_HANDLE_BASE 0xcafe0000u

NvHandle aether_next_handle(aether_device *d) { return d->nextHandle++; }

/* Record which step failed before unwinding. Every `goto fail` goes through
 * this, so there is no path out of open() that cannot say what happened. */
#define FAIL(stage, status)                                                    \
  do {                                                                         \
    d->failStage = (stage);                                                    \
    d->failStatus = (int)(status);                                             \
    goto fail;                                                                 \
  } while (0)

int aether_device_open(aether_device *d, int index) {
  memset(d, 0, sizeof *d);
  d->ctlFd = -1;
  d->gpuFd = -1;
  d->attachFd = -1;
  d->index = index;
  d->nextHandle = AETHER_HANDLE_BASE;

  d->ctlFd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
  if (d->ctlFd < 0) return -1;

  /*
   * The initialisation handshake, in the order a working driver performs it.
   * Captured by tracing opens and ioctls of a Vulkan compute submit:
   *
   *   open /dev/nvidiactl                    -> primary control fd
   *   CHECK_VERSION_STR, SYS_PARAMS, CARD_INFO on it
   *   open /dev/nvidiaN, REGISTER_FD on each
   *   open a SECOND /dev/nvidiactl           -> and ATTACH_GPUS_TO_FD on THAT
   *
   * The last step is the one that is easy to get wrong and produces no useful
   * error: ATTACH_GPUS_TO_FD must go on a FRESH control fd, as its first
   * operation, with no REGISTER_FD first. Issued on a device fd -- which is
   * what the name suggests -- it returns EINVAL every time.
   */
  {
    struct { NvU32 cmd, reply; char v[64]; } ver;
    memset(&ver, 0, sizeof ver);
    ver.cmd = '1'; /* NV_RM_API_VERSION_CMD_RELAXED */
    aether_ioctl(d->ctlFd, NV_ESC_CHECK_VERSION_STR, &ver, sizeof ver);

    /*
     * CARD_INFO is an array of nv_ioctl_card_info_t, whose layout follows from
     * kernel-open/common/inc/nv-ioctl.h under natural alignment:
     *
     *   NvBool        valid           @0   (NvBool is NvU8)
     *   nv_pci_info_t pci_info        @4   (12 bytes: u32,u8,u8,u8,pad,u16,u16)
     *   NvU32         gpu_id          @16
     *   NvU16         interrupt_line  @20
     *   NvU64         reg_address     @24  (NV_ALIGN_BYTES(8))
     *   NvU64         reg_size        @32
     *   NvU64         fb_address      @40
     *   NvU64         fb_size         @48
     *   NvU32         minor_number    @56
     *   NvU8          dev_name[10]    @60
     *                                 -> 72 bytes with 8-byte struct alignment
     *
     * TWO FIELDS MATTER AND BOTH WERE BEING IGNORED. `valid` says whether the
     * slot holds a card at all, and `minor_number` is the N in /dev/nvidiaN.
     *
     * WHY THIS IS NOT A DETAIL: the first version assumed index == minor and
     * opened /dev/nvidia0. That worked on one rented pod and silently failed on
     * the next, which was handed a GPU at /dev/nvidia1 -- open() returned
     * ENOENT, the device reported "no GPU", and the hardware test SKIPPED
     * rather than failed. A test that skips when the environment surprises it
     * is a test that tells you nothing, so the enumeration has to be real.
     */
    /* NV_MAX_DEVICES entries of 72 bytes. The size is part of the ioctl request
     * code, so it has to be exactly what the kernel expects. (nvlimits.h) */
    static unsigned char ci[72 * 32];
    memset(ci, 0, sizeof ci);
    if (aether_ioctl(d->ctlFd, NV_ESC_CARD_INFO, ci, sizeof ci) != 0)
      FAIL("CARD_INFO", errno);

    /*
     * "The index-th valid card" is still the wrong selector, and hardware said
     * so: inside a RunPod container CARD_INFO reports TWO valid cards, minors 0
     * and 1, while only /dev/nvidia1 exists in the container's device namespace.
     * RM's card list is host-wide; the device nodes we are allowed to open are
     * not. Enumerating the list alone picks a GPU this process cannot touch.
     *
     * So the selector is "the index-th card we can actually OPEN". Trying the
     * open is the only way to know, and it is also exactly the property the
     * caller cares about.
     */
    int seen = -1;
    for (unsigned e = 0; (e + 1) * 72u <= sizeof ci; e++) {
      const unsigned char *card = ci + e * 72u;
      if (!card[0]) continue; /* valid == 0: an empty slot, not a numbering gap */

      const int minor = (int)*(const NvU32 *)(card + 56);
      char path[32];
      snprintf(path, sizeof path, "/dev/nvidia%d", minor);
      int fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd < 0) continue; /* present to RM, absent to us */

      if (++seen != index) { close(fd); continue; }
      d->gpuId = *(const NvU32 *)(card + 16);
      d->minor = minor;
      d->gpuFd = fd;
      break;
    }
    if (d->gpuFd < 0) FAIL("no openable GPU at that index", seen + 1);
  }

  aether_register_fd(d, d->gpuFd);

  /* Attach the GPU on its own fresh control fd. Held open for the lifetime of
   * the device: closing it would detach. */
  if (d->gpuId) {
    d->attachFd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (d->attachFd >= 0) {
      static NvU32 gpus[32];
      memset(gpus, 0, sizeof gpus);
      gpus[0] = d->gpuId;
      aether_ioctl(d->attachFd, NV_ESC_ATTACH_GPUS_TO_FD, gpus, sizeof gpus);
    }
  }

  /* The client is the root of our handle namespace. It is allocated with
   * itself as both root and parent, which is the one place the chain is
   * self-referential — RM special-cases it. */
  {
    NVOS21_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hRoot = 0;
    p.hObjectParent = 0;
    p.hObjectNew = aether_next_handle(d);
    p.hClass = NV01_ROOT_CLIENT;
    if (aether_ioctl(d->ctlFd, NV_ESC_RM_ALLOC, &p, sizeof p) < 0)
      FAIL("NV01_ROOT_CLIENT ioctl", errno);
    if (p.status != NV_OK) FAIL("NV01_ROOT_CLIENT", p.status);
    d->client = p.hObjectNew;
  }

  /*
   * The device instance, which is NOT the card index and NOT the minor number.
   *
   * NV0080_ALLOC_PARAMETERS.deviceId is RM's device instance, assigned in
   * attach order across the whole host. Inside a container that was handed one
   * GPU out of several, it can be any value -- so ask rather than assume.
   * NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2 translates the gpu_id we already have
   * from CARD_INFO into exactly this number. (ctrl0000gpu.h)
   */
  NvU32 deviceInstance;
  {
    struct {
      NvU32 gpuId, gpuFlags, deviceInstance, subDeviceInstance;
      NvU32 sliStatus, boardId, gpuInstance;
      NvS32 numaId;
    } info;
    memset(&info, 0, sizeof info);
    info.gpuId = d->gpuId;
    int rc = aether_control(d, d->client, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2,
                            &info, sizeof info);
    if (rc != 0) FAIL("GPU_GET_ID_INFO_V2", rc);
    deviceInstance = info.deviceInstance;
  }

  {
    NV0080_ALLOC_PARAMETERS dp;
    memset(&dp, 0, sizeof dp);
    dp.deviceId = deviceInstance;
    dp.hClientShare = d->client;
    int rc = aether_alloc(d, d->client, &d->device, NV01_DEVICE_0, &dp, sizeof dp);
    if (rc != 0) FAIL("NV01_DEVICE_0", rc);
  }

  {
    NV2080_ALLOC_PARAMETERS sp;
    memset(&sp, 0, sizeof sp);
    sp.subDeviceId = 0;
    int rc = aether_alloc(d, d->device, &d->subdevice, NV20_SUBDEVICE_0, &sp,
                          sizeof sp);
    if (rc != 0) FAIL("NV20_SUBDEVICE_0", rc);
  }

  {
    /* A zero vaSize asks RM for its default address space, which is what we
     * want: we are not trying to control placement, only to have a space that
     * buffers can be mapped into. */
    NV_VASPACE_ALLOCATION_PARAMETERS vp;
    memset(&vp, 0, sizeof vp);
    int rc = aether_alloc(d, d->device, &d->vaspace, FERMI_VASPACE_A, &vp,
                          sizeof vp);
    if (rc != 0) FAIL("FERMI_VASPACE_A", rc);
  }

  return 0;

fail:
  {
    /* close() zeroes the struct, and the diagnosis must survive it -- otherwise
     * the caller gets a clean, silent, uninformative failure. */
    const char *stage = d->failStage;
    const int status = d->failStatus;
    aether_device_close(d);
    d->failStage = stage;
    d->failStatus = status;
  }
  return -1;
}

void aether_device_close(aether_device *d) {
  /* Free in reverse order of creation: children before parents. RM would tear
   * down the subtree on client free anyway, but relying on that hides leaks in
   * any path that frees objects individually. */
  if (d->vaspace) aether_free(d, d->vaspace);
  if (d->subdevice) aether_free(d, d->subdevice);
  if (d->device) aether_free(d, d->device);
  if (d->client) aether_free(d, d->client);
  if (d->attachFd >= 0) close(d->attachFd);
  if (d->gpuFd >= 0) close(d->gpuFd);
  if (d->ctlFd >= 0) close(d->ctlFd);
  memset(d, 0, sizeof *d);
  d->ctlFd = -1;
  d->gpuFd = -1;
  d->attachFd = -1;
}

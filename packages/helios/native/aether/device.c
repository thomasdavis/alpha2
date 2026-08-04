/*
 * device.c — see device.h.
 */
/* O_CLOEXEC is POSIX-2008, not C11, so the build uses -std=gnu11. A
 * _GNU_SOURCE define here would be a no-op: it has to precede the first system
 * header, and device.h pulls one in. Putting the requirement in the build flags
 * rather than in a define that silently does nothing. */
#include "device.h"
#include "ioctl.h"

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

int aether_alloc(aether_device *d, NvHandle parent, NvHandle *out, NvV32 cls,
                 void *params, NvU32 paramsSize) {
  NVOS21_PARAMETERS p;
  memset(&p, 0, sizeof p);
  p.hRoot = d->client;
  p.hObjectParent = parent;
  p.hObjectNew = aether_next_handle(d);
  p.hClass = cls;
  p.pAllocParms = (NvP64)(uintptr_t)params;
  p.paramsSize = paramsSize;

  int rc = aether_ioctl(d->ctlFd, NV_ESC_RM_ALLOC, &p, sizeof p);
  if (rc < 0) return rc;
  /* The ioctl succeeding says nothing about RM accepting the request. */
  if (p.status != NV_OK) return (int)p.status;

  *out = p.hObjectNew;
  return 0;
}

int aether_free(aether_device *d, NvHandle object) {
  /* NV_ESC_RM_FREE reuses the alloc parameter block, with hObjectNew naming
   * the object to release. */
  NVOS21_PARAMETERS p;
  memset(&p, 0, sizeof p);
  p.hRoot = d->client;
  p.hObjectParent = d->client;
  p.hObjectNew = object;

  int rc = aether_ioctl(d->ctlFd, NV_ESC_RM_FREE, &p, sizeof p);
  if (rc < 0) return rc;
  return p.status == NV_OK ? 0 : (int)p.status;
}

int aether_control(aether_device *d, NvHandle object, NvV32 cmd, void *params,
                   NvU32 paramsSize) {
  NVOS54_PARAMETERS p;
  memset(&p, 0, sizeof p);
  p.hClient = d->client;
  p.hObject = object;
  p.cmd = cmd;
  p.params = (NvP64)(uintptr_t)params;
  p.paramsSize = paramsSize;

  int rc = aether_ioctl(d->ctlFd, NV_ESC_RM_CONTROL, &p, sizeof p);
  if (rc < 0) return rc;
  return p.status == NV_OK ? 0 : (int)p.status;
}

int aether_device_open(aether_device *d, int index) {
  memset(d, 0, sizeof *d);
  d->ctlFd = -1;
  d->gpuFd = -1;
  d->index = index;
  d->nextHandle = AETHER_HANDLE_BASE;

  d->ctlFd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
  if (d->ctlFd < 0) return -1;

  char path[32];
  snprintf(path, sizeof path, "/dev/nvidia%d", index);
  d->gpuFd = open(path, O_RDWR | O_CLOEXEC);
  if (d->gpuFd < 0) goto fail;

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
    if (aether_ioctl(d->ctlFd, NV_ESC_RM_ALLOC, &p, sizeof p) < 0) goto fail;
    if (p.status != NV_OK) goto fail;
    d->client = p.hObjectNew;
  }

  {
    NV0080_ALLOC_PARAMETERS dp;
    memset(&dp, 0, sizeof dp);
    dp.deviceId = (NvU32)index;
    dp.hClientShare = d->client;
    if (aether_alloc(d, d->client, &d->device, NV01_DEVICE_0, &dp, sizeof dp) != 0)
      goto fail;
  }

  {
    NV2080_ALLOC_PARAMETERS sp;
    memset(&sp, 0, sizeof sp);
    sp.subDeviceId = 0;
    if (aether_alloc(d, d->device, &d->subdevice, NV20_SUBDEVICE_0, &sp,
                     sizeof sp) != 0)
      goto fail;
  }

  {
    /* A zero vaSize asks RM for its default address space, which is what we
     * want: we are not trying to control placement, only to have a space that
     * buffers can be mapped into. */
    NV_VASPACE_ALLOCATION_PARAMETERS vp;
    memset(&vp, 0, sizeof vp);
    if (aether_alloc(d, d->device, &d->vaspace, FERMI_VASPACE_A, &vp,
                     sizeof vp) != 0)
      goto fail;
  }

  return 0;

fail:
  aether_device_close(d);
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
  if (d->gpuFd >= 0) close(d->gpuFd);
  if (d->ctlFd >= 0) close(d->ctlFd);
  memset(d, 0, sizeof *d);
  d->ctlFd = -1;
  d->gpuFd = -1;
}

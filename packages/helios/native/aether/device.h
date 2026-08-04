/*
 * device.h — opening the GPU and building the RM object chain.
 *
 * WHAT: opens the character devices and allocates the four objects every
 * subsequent call needs: client, device, subdevice, address space.
 *
 * WHY these four and in this order: RM is a hierarchical object model, not a
 * flat API. Nothing can be allocated without a parent, so the chain has to
 * exist before memory or channels do:
 *
 *     client   (NV01_ROOT_CLIENT)   our handle namespace
 *       device (NV01_DEVICE_0)      one physical GPU
 *         subdevice (NV20_SUBDEVICE_0)  one GPU within it — where queries live
 *         vaspace   (FERMI_VASPACE_A)   the GPU virtual address space
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no memory allocation, no channels. Those
 * are Gaia and Hermes. This file gets us to the point where those are possible
 * and no further.
 *
 * On handles: for most objects RM does not hand them out -- the caller invents
 * them, and they only have to be unique within a client. We allocate
 * sequentially from a counter (aether_next_handle) because a colliding handle
 * is an NV_ERR_INSERT_DUPLICATE_NAME at best and an aliased object at worst.
 *
 * THE ROOT CLIENT IS THE EXCEPTION, and hardware taught us this rather than the
 * headers. Allocating NV01_ROOT_CLIENT with hObjectNew = 0xcafe0000 returns
 * NV_OK but with hObjectNew REWRITTEN to something of RM's choosing -- observed
 * 0xc1d71ea9 on an RTX 3070. So the client handle must be read back out of the
 * parameter block after the call, never assumed. Code that trusted its
 * requested value would address a non-existent object on every subsequent call,
 * and would do so without any error at the point of the mistake.
 */
#ifndef HELIOS_AETHER_DEVICE_H
#define HELIOS_AETHER_DEVICE_H

#include "nv_abi.h"

typedef struct {
  int ctlFd;  /* /dev/nvidiactl — the control channel, all RM escapes go here */
  int gpuFd;  /* /dev/nvidiaN  — needed for mapping this GPU's memory */
  int index;  /* which GPU */

  NvHandle client;
  NvHandle device;
  NvHandle subdevice;
  NvHandle vaspace;

  NvU32 nextHandle; /* our own handle allocator */
} aether_device;

/*
 * Open GPU `index` and build the object chain. Returns 0 on success, or a
 * negative errno / positive NV_ERR_* status.
 *
 * On failure the partially built chain is torn down, so a failed open never
 * leaks kernel objects — RM objects outlive the process only if the fd stays
 * open, but a half-built device is still a bug we do not want to debug later.
 */
int aether_device_open(aether_device *d, int index);

void aether_device_close(aether_device *d);

/*
 * The driver initialisation handshake, performed before any RM object exists.
 *
 * A working driver issues CHECK_VERSION_STR and CARD_INFO on the control fd
 * before touching RM. The version check is a real handshake: it takes the
 * running kernel module's version string and answers RECOGNIZED or not.
 *
 * Verified against driver 580.95.05 -- reply == 1 (RECOGNIZED).
 *
 * NOT yet working: NV_ESC_ATTACH_GPUS_TO_FD, which a working driver also issues
 * and which returns EINVAL for every payload shape tried so far (bare id array,
 * count-then-ids, on either fd). Its gpu_id comes from CARD_INFO offset 16
 * (0x00000a00 on this machine, alongside PCI 10de:2484). This is the largest
 * remaining known gap between our init sequence and a working one.
 */
int aether_check_version(aether_device *d, const char *driverVersion);

/*
 * Associate a secondary file descriptor with the control fd.
 *
 * DIRECTION MATTERS and is the opposite of what the name suggests: the ioctl is
 * issued ON THE NEW FD, naming the control fd it should be attached to. Issued
 * the other way round it returns EINVAL.
 *
 *   aether_register_fd(d, newfd)  ->  ioctl(newfd, NV_ESC_REGISTER_FD, &ctlFd)
 *
 * A working CUDA process does this for every extra descriptor it opens. Ours
 * were unregistered, which is now fixed -- though it did not by itself make the
 * GPU consume submitted work.
 */
int aether_register_fd(aether_device *d, int fd);

/* Allocate a fresh RM handle within this client. */
NvHandle aether_next_handle(aether_device *d);

/* Allocate an RM object. `params` may be NULL for classes that take none. */
int aether_alloc(aether_device *d, NvHandle parent, NvHandle *out, NvV32 cls,
                 void *params, NvU32 paramsSize);

/* Free an RM object. */
int aether_free(aether_device *d, NvHandle object);

/* Invoke a control command on an object. */
int aether_control(aether_device *d, NvHandle object, NvV32 cmd, void *params,
                   NvU32 paramsSize);

#endif /* HELIOS_AETHER_DEVICE_H */

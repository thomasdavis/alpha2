/*
 * rm.c — the RM object verbs.
 *
 * WHAT: allocate, free and control RM objects, plus the two fd-level escapes
 * that have to happen before any of that works.
 *
 * WHY these are separate from device.c: they are the vocabulary every layer
 * above uses, and they know nothing about which GPU they are talking to. Device
 * lifecycle -- enumeration, opening nodes, building the object chain -- is a
 * different concern that happens once. Splitting them also keeps each file
 * inside the size cap without inventing an artificial boundary.
 *
 * THE RECURRING LESSON, and the reason every one of these checks status
 * separately from the ioctl return: NV_OK from RM means "the request was
 * accepted", not "you now have what you asked for". A successful ioctl carrying
 * a failed status is the normal way these calls fail.
 */
#include "device.h"
#include "ioctl.h"

#include <stdio.h>
#include <string.h>

int aether_check_version(aether_device *d, const char *driverVersion) {
  /* nv_ioctl_rm_api_version_t — kernel-open/common/inc/nv-ioctl.h.
   * cmd '1' is NV_RM_API_VERSION_CMD_RELAXED; reply 1 is RECOGNIZED. */
  struct {
    NvU32 cmd, reply;
    char versionString[64];
  } v;
  memset(&v, 0, sizeof v);
  v.cmd = '1';
  snprintf(v.versionString, sizeof v.versionString, "%s", driverVersion);

  int rc = aether_ioctl(d->ctlFd, NV_ESC_CHECK_VERSION_STR, &v, sizeof v);
  if (rc < 0) return rc;
  return v.reply == 1 ? 0 : -1;
}

int aether_register_fd(aether_device *d, int fd) {
  /* Issued on the NEW fd, naming the control fd — see the header. */
  int arg = d->ctlFd;
  return aether_ioctl(fd, NV_ESC_REGISTER_FD, &arg, sizeof arg);
}

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


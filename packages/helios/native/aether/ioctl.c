/*
 * ioctl.c — see ioctl.h.
 *
 * WHAT: the syscall, and a status-code name table.
 *
 * WHY: isolated in its own translation unit so that "how many syscalls did we
 * make" is answerable by putting a counter in exactly one place, and so the
 * layers above cannot reach ioctl() at all (standard 4).
 */
#include "ioctl.h"
#include "nv_abi.h"

#include <errno.h>
#include <stdio.h>
#include <sys/ioctl.h>

int aether_ioctl(int fd, unsigned nr, void *params, size_t size) {
  /* RM encodes the struct size into the request code; a mismatch between this
   * and the struct we pass is rejected by the kernel rather than silently
   * truncated, which is the behaviour we want. */
  unsigned long request = AE_IOWR(nr, size);

  int rc;
  do {
    rc = ioctl(fd, request, params);
  } while (rc < 0 && errno == EINTR); /* the only retry: a signal, not a fault */

  if (rc < 0) return -errno;
  return 0;
}

const char *aether_status_name(unsigned status) {
  switch (status) {
    case NV_OK: return "NV_OK";
    case NV_ERR_INVALID_ARGUMENT: return "NV_ERR_INVALID_ARGUMENT";
    case NV_ERR_INVALID_STATE: return "NV_ERR_INVALID_STATE";
    case NV_ERR_NOT_SUPPORTED: return "NV_ERR_NOT_SUPPORTED";
    case NV_ERR_INSUFFICIENT_RESOURCES: return "NV_ERR_INSUFFICIENT_RESOURCES";
    case NV_ERR_NO_MEMORY: return "NV_ERR_NO_MEMORY";
    default: {
      /* Deliberately not a fallthrough to "unknown": the numeric value is the
       * only thing that lets someone grep nvstatuscodes.h for the real name. */
      static char buf[32];
      snprintf(buf, sizeof buf, "NV_ERR_0x%08x", status);
      return buf;
    }
  }
}

/*
 * ioctl.h — the single place a syscall crosses into the kernel.
 *
 * WHAT: request-code assembly and the one wrapper every RM call goes through.
 *
 * WHY: RM's ioctl requests encode the parameter struct's size in the request
 * itself, so the code and the struct must agree or the kernel rejects the call.
 * Deriving the request from sizeof(T) at the call site makes that impossible to
 * get wrong, and gives us exactly one line to instrument when a call misbehaves.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no retry, no interpretation of RM status.
 * A short read here is a transport failure; an RM error is a `status` field
 * inside the parameter struct and is the caller's business.
 *
 * PROVENANCE: the _IOC layout is Linux's own (asm-generic/ioctl.h). RM uses
 * NV_IOCTL_MAGIC 'F' — nv-linux.h / nv_escape.h.
 */
#ifndef HELIOS_AETHER_IOCTL_H
#define HELIOS_AETHER_IOCTL_H

#include <stddef.h>

/* nv-linux.h: #define NV_IOCTL_MAGIC 'F' */
#define NV_IOCTL_MAGIC 'F'

/* Linux _IOC field widths (asm-generic/ioctl.h). Restated rather than included
 * so this header stands alone and the bit layout is visible at the point of
 * use — it is the thing most likely to be wrong on a new platform. */
#define AE_IOC_NRBITS 8
#define AE_IOC_TYPEBITS 8
#define AE_IOC_SIZEBITS 14
#define AE_IOC_NRSHIFT 0
#define AE_IOC_TYPESHIFT (AE_IOC_NRSHIFT + AE_IOC_NRBITS)
#define AE_IOC_SIZESHIFT (AE_IOC_TYPESHIFT + AE_IOC_TYPEBITS)
#define AE_IOC_DIRSHIFT (AE_IOC_SIZESHIFT + AE_IOC_SIZEBITS)

/* Direction bits are from the *userspace* point of view: READ means the kernel
 * writes back into our struct. Every RM escape is read-write, because status is
 * returned in the same struct that carried the request. */
#define AE_IOC_WRITE 1U
#define AE_IOC_READ 2U

#define AE_IOC(dir, type, nr, size)                                            \
  (((unsigned)(dir) << AE_IOC_DIRSHIFT) |                                      \
   ((unsigned)(type) << AE_IOC_TYPESHIFT) |                                    \
   ((unsigned)(nr) << AE_IOC_NRSHIFT) | ((unsigned)(size) << AE_IOC_SIZESHIFT))

/* The only form RM uses. */
#define AE_IOWR(nr, size)                                                      \
  AE_IOC(AE_IOC_READ | AE_IOC_WRITE, NV_IOCTL_MAGIC, (nr), (size))

/*
 * Issue one RM escape. Returns 0 on a successful syscall, -errno otherwise.
 *
 * NOTE the two failure levels: a 0 return means the *ioctl* succeeded, which
 * says nothing about whether RM liked the request. The caller must still check
 * the `status` field of its parameter struct. Conflating those two is the
 * classic way to build a driver that appears to work.
 */
int aether_ioctl(int fd, unsigned nr, void *params, size_t size);

/* Human-readable RM status, for error paths. Returns a static string. */
const char *aether_status_name(unsigned status);

#endif /* HELIOS_AETHER_IOCTL_H */

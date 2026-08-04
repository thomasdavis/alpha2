/*
 * spy.h — shared surface between the QMD spy and its pushbuffer decoder.
 *
 * These are development tools, outside the training loop, in the same category
 * as nvdisasm. Nothing here is linked into anything that ships.
 */
#ifndef HELIOS_TOOLS_SPY_H
#define HELIOS_TOOLS_SPY_H

#include <stdint.h>
#include <stdio.h>

extern FILE *spy_log;

/* Data words following a header, given its opcode; negative means "stop", which
 * covers END_PB_SEGMENT and anything unrecognised. */
int spy_data_words(uint32_t op, uint32_t count);

/* Decode and print the method stream at `addr`, which must be host-readable. */
void spy_dump_pushbuffer(uint64_t addr, uint32_t dwords);

/* The name of a compute-class method, or NULL. Exposed because the QMD scanner
 * annotates raw method streams with it, and a second copy of the table is a
 * second thing to keep in step with the hardware headers. */
const char *spy_method_name(uint32_t method);

/*
 * Address-space bookkeeping, in spy_memory.c.
 *
 * `L` is the log stream, shared because every part of the spy writes to the
 * same file and threading a handle through a scanner that runs on its own
 * thread would buy nothing.
 */
#include <stdio.h>
extern FILE *L;

/* The GPU-visible mappings, as read from /proc/self/maps. The scanner walks
 * these directly, so they are exposed rather than hidden behind an accessor
 * that would only wrap a loop. */
/* How many mappings the scanner will track. Generous: a CUDA process maps a few
 * dozen, and running out silently would make the scanner miss a region rather
 * than report that it did. */
/* The band CUDA maps GPU buffers into. Deliberately generous: precision comes
 * from the consequence check, not from this. */
#define GPU_VA_LO 0x100000000ull
#define GPU_VA_HI 0x1000000000ull

#define MAX_REGIONS 512
struct spy_region { uint64_t lo, hi; };
extern struct spy_region spy_regions[];
extern int spy_nregions;
void spy_load_regions(void);
int spy_in_gpu_region(uint64_t addr, uint64_t len);
long spy_read_self(uint64_t addr, void *dst, size_t len);

#endif /* HELIOS_TOOLS_SPY_H */

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

#endif /* HELIOS_TOOLS_SPY_H */

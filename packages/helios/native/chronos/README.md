# chronos — reserved, and currently empty

`build-stack.mjs` lists seven layers and this is the one with no source in it:

```
aether       ioctl transport
gaia         memory and address space
hermes       channels, pushbuffer, launch
chronos      fences and timeline     <- nothing here
hephaestus   SASS assembler
prometheus   kernel IR and codegen
helios       the facade
```

**That is deliberate now, and it was not before.** An empty directory in a
declared architecture reads as "unfinished" to one person and "unnecessary" to
the next, and both of them are guessing. So:

## Where the timeline actually lives

There is exactly one synchronisation primitive in the stack and it is a
**release semaphore written by the pushbuffer, polled by the host**:

* `hermes_semaphore_release()` (hermes/pushbuffer.c) appends the release to the
  pushbuffer, so it retires in order behind the launches before it;
* `helios_context.fence` (helios/context.h) is the buffer it writes to and
  `fenceValue` the counter it writes, incremented per launch so two waits cannot
  alias;
* the host **spins** on that word. `stats().spinNs` is how long, and it is the
  number the whole performance program is written against — "GPU 74.7 ms/step =
  94% of wall" is that counter against the wall clock.

So the layer's job is done, one layer down, in about forty lines.

## What would move here, and what it would be worth

A `chronos/` worth building is one that makes the host **stop spinning**, and
each of these is a real thing this stack does not have:

| capability | what it needs | what it is worth |
|---|---|---|
| **Blocking wait** on a fence | the RM event/notifier ioctls in `aether` — `NV01_EVENT_OS_EVENT` and a poll/epoll on the returned fd | the host burns a core spinning today. Costs nothing at 94% GPU; costs a whole core when the step gets short, which is the direction the program is going |
| **Multiple in-flight fences** | a ring of fence slots rather than one word, so a wait can name a point rather than "everything" | the launch barrier is conservative — every buffer counts as read AND written, so 69% of launches wait for a full drain. A finer rule was counted and is worth ~1%, but that measurement assumed one fence; with several, "wait for the launch that wrote MY input" becomes expressible |
| **Timestamps in the pushbuffer** | the semaphore release form that writes a GPU clock value | per-kernel GPU time WITHOUT draining. Every profiler in packages/tests drains per op to attribute time, which removes all overlap and inflates every row by ~10 us a call — and has now misled this stack three times. This is the fix for the instrument, not for the kernel |

The third one is the one to build first if any of them is built, and it is worth
saying why: it is not an optimisation at all, it is the measurement device that
the optimisations are being chosen with. `profile-gpu-by-op.mjs` currently
reports a drained total 1.23x larger than the real step and has to warn the
reader about it in its own output.

## Until then

Do not add files here to make the layer look inhabited. `LAYERS` is ordered
bottom-up and the index in that array **is** the dependency rule — a layer's
test binary links that layer and everything below it, so the empty entry costs
nothing and keeps the position reserved. The build already tolerates it:
`sources()` returns `[]` for a directory that does not exist, and `build()`
returns early when there is no `chronos_test.c`.

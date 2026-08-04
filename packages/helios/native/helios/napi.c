/*
 * napi.c — the boundary, and the only file here that knows JavaScript exists.
 *
 * WHAT: context lifetime, tensor allocation, and the zero-copy view that lets
 * JavaScript write into device-visible memory directly. The operations
 * themselves are in napi_ops.c.
 *
 * THE ZERO-COPY VIEW IS THE POINT. A tensor's host mapping is handed to
 * JavaScript as an external ArrayBuffer, so a Float32Array over it writes
 * straight into the pages the GPU reads. There is no upload, no staging buffer,
 * and no copy in either direction -- which is the marshalling the whole rewrite
 * exists to delete. The measured cost it replaces was 23% of a step spent
 * packing dispatch records into ArrayBuffers.
 *
 * WHAT THAT COSTS, stated because it is a real hazard: the buffer stays valid
 * only while the tensor does. JavaScript holding a view past the free reads
 * memory the pool has handed to someone else. The handle is generation-checked
 * and the VIEW is not -- a check on every element access would defeat the
 * purpose of not copying. So freeing a tensor whose view is still live is a
 * caller error, and the TypeScript side is where that has to be prevented.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no threads, no async work queue, no
 * promises. Every call runs on the JavaScript thread and returns when the GPU
 * is done. That blocks, and it is honest: the alternative is a queue whose
 * completion order has to be reasoned about before anything is even correct.
 */
#include <node_api.h>

#include "dispatch.h"

#include <stdio.h>
#include <string.h>

/*
 * One context per process.
 *
 * Not per-instance: the device, the channel and the program cache are all
 * process-wide by nature, and letting JavaScript make two would mean two
 * channels racing over one constant bank. If a second device is ever wanted it
 * will be a second context here, not a second instance of this module.
 */
static helios_context g_ctx;
static int g_open;

napi_value hl_napi_register_ops(napi_env env, napi_value exports);

/* Read argument `i` as a uint32, or 0. Out-of-range indices give 0 rather than
 * reading past the array, so a call with too few arguments fails in the
 * dispatcher rather than here. */
NvU32 hl_arg_u32(napi_env env, napi_callback_info info, size_t i) {
  size_t argc = 16;
  napi_value argv[16];
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  if (i >= argc) return 0;
  NvU32 v = 0;
  napi_get_value_uint32(env, argv[i], &v);
  return v;
}

double hl_arg_double(napi_env env, napi_callback_info info, size_t i) {
  size_t argc = 16;
  napi_value argv[16];
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  if (i >= argc) return 0;
  double v = 0;
  napi_get_value_double(env, argv[i], &v);
  return v;
}

helios_context *hl_context(void) { return g_open ? &g_ctx : NULL; }

napi_value hl_result(napi_env env, int rc) {
  napi_value out;
  napi_get_boolean(env, rc == 0, &out);
  return out;
}

static napi_value js_open(napi_env env, napi_callback_info info) {
  napi_value out;
  if (g_open) {
    napi_get_boolean(env, true, &out);
    return out;
  }
  const NvU32 index = hl_arg_u32(env, info, 0);
  if (helios_context_open(&g_ctx, (int)index) != 0) {
    /* Throwing rather than returning false: a caller that ignores a false here
     * goes on to allocate tensors against a dead context, and every subsequent
     * failure points somewhere unrelated. */
    char msg[160];
    snprintf(msg, sizeof msg, "helios: device open failed at %s",
             g_ctx.failStage ? g_ctx.failStage : "unknown");
    napi_throw_error(env, NULL, msg);
    napi_get_boolean(env, false, &out);
    return out;
  }
  g_open = 1;
  napi_get_boolean(env, true, &out);
  return out;
}

static napi_value js_close(napi_env env, napi_callback_info info) {
  (void)info;
  if (g_open) {
    helios_tensor_release_all(&g_ctx);
    helios_context_close(&g_ctx);
    g_open = 0;
  }
  napi_value out;
  napi_get_undefined(env, &out);
  return out;
}

static napi_value js_alloc(napi_env env, napi_callback_info info) {
  napi_value out;
  if (!g_open) {
    napi_throw_error(env, NULL, "helios: alloc before open");
    napi_get_undefined(env, &out);
    return out;
  }
  const NvU32 bytes = hl_arg_u32(env, info, 0);
  const helios_tensor t = helios_tensor_alloc(&g_ctx, bytes);
  napi_create_uint32(env, t, &out);
  return out;
}

static napi_value js_free(napi_env env, napi_callback_info info) {
  helios_tensor_free(hl_arg_u32(env, info, 0));
  napi_value out;
  napi_get_undefined(env, &out);
  return out;
}

/*
 * The host mapping, as an ArrayBuffer JavaScript can write through.
 *
 * The finalizer is NULL and deliberately so: this memory belongs to the pool,
 * not to the garbage collector, and letting V8 free it would return pages to
 * the allocator while the GPU still has them mapped. The tensor's lifetime is
 * the handle's, and nothing about the view changes that.
 */
static napi_value js_view(napi_env env, napi_callback_info info) {
  napi_value out;
  const helios_tensor t = hl_arg_u32(env, info, 0);
  void *host = helios_tensor_host(t);
  if (!host) {
    napi_get_null(env, &out);
    return out;
  }
  napi_create_external_arraybuffer(env, host, (size_t)helios_tensor_bytes(t),
                                   NULL, NULL, &out);
  return out;
}

static napi_value js_stats(napi_env env, napi_callback_info info) {
  (void)info;
  const helios_tensor_stats s = helios_tensor_get_stats();
  napi_value out, v;
  napi_create_object(env, &out);
  napi_create_uint32(env, s.live, &v);
  napi_set_named_property(env, out, "live", v);
  napi_create_uint32(env, s.pooled, &v);
  napi_set_named_property(env, out, "pooled", v);
  napi_create_uint32(env, s.allocations, &v);
  napi_set_named_property(env, out, "allocations", v);
  napi_create_uint32(env, helios_program_count(), &v);
  napi_set_named_property(env, out, "programs", v);
  return out;
}

void hl_export(napi_env env, napi_value exports, const char *name,
               napi_callback fn) {
  napi_value f;
  napi_create_function(env, name, NAPI_AUTO_LENGTH, fn, NULL, &f);
  napi_set_named_property(env, exports, name, f);
}

static napi_value init(napi_env env, napi_value exports) {
  hl_export(env, exports, "open", js_open);
  hl_export(env, exports, "close", js_close);
  hl_export(env, exports, "alloc", js_alloc);
  hl_export(env, exports, "free", js_free);
  hl_export(env, exports, "view", js_view);
  hl_export(env, exports, "stats", js_stats);
  return hl_napi_register_ops(env, exports);
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)

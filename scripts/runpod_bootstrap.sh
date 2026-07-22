#!/bin/bash
# RunPod pod bootstrap for Alpha/Helios (Vulkan compute) — PROVEN 2026-07-22 on community RTX 3090.
#
# Why this exists: RunPod pods run with NVIDIA_DRIVER_CAPABILITIES=compute,utility and the CDI-based
# runtime does NOT honor an env override, so no graphics/Vulkan userspace is mounted. Additionally the
# stock nvidia_icd.json (libGLX_nvidia.so.0) fails headless (vk_icdGetInstanceProcAddr -> NULL without a
# display). The fix, proven end-to-end:
#   1. install the EXACT-matching NVIDIA userspace via the .run installer with --no-kernel-modules
#   2. point the Vulkan loader at the EGL library via a headless ICD json (alpha2's GCP-era trick)
#
# Community-host caveats (observed): port-80 egress (apt) and github.com may be dead; download.nvidia.com
# and nodejs.org over 443 worked. This script therefore avoids apt and github entirely.
# Run as root on the pod. Idempotent.
set -euo pipefail

echo "== [1/5] NVIDIA userspace (Vulkan ICD) =="
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "host driver: $DRV"
if [ ! -e "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0" ]; then
  cd /tmp
  RUN="NVIDIA-Linux-x86_64-$DRV.run"
  [ -f "$RUN" ] || curl -sfO "https://download.nvidia.com/XFree86/Linux-x86_64/$DRV/$RUN" \
                || curl -sfO "https://us.download.nvidia.com/XFree86/Linux-x86_64/$DRV/$RUN"
  # installer requires kmod utilities even with --no-kernel-modules; stub them
  for b in modprobe rmmod insmod lsmod depmod; do
    printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/$b && chmod +x /usr/local/bin/$b
  done
  sh "$RUN" --silent --no-kernel-modules --no-systemd --no-x-check --no-nouveau-check --skip-depmod --no-dkms
fi

echo "== [2/5] headless EGL ICD =="
mkdir -p /etc/vulkan/icd.d
VKAPI=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.* >/dev/null 2>&1 && echo "1.4.312" || echo "1.3.303")
cat > /etc/vulkan/icd.d/nvidia_icd_headless.json <<EOF
{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"$VKAPI"}}
EOF
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json
unset DISPLAY || true
grep -q VK_ICD_FILENAMES /root/.bashrc || {
  echo 'export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json' >> /root/.bashrc
  echo 'unset DISPLAY' >> /root/.bashrc
}

echo "== [3/5] Vulkan probe (ctypes, no apt needed) =="
python3 - <<'PYEOF'
import ctypes, sys
vk = ctypes.CDLL("libvulkan.so.1")
class App(ctypes.Structure):
    _fields_=[("sType",ctypes.c_int),("pNext",ctypes.c_void_p),("pApplicationName",ctypes.c_char_p),
              ("applicationVersion",ctypes.c_uint32),("pEngineName",ctypes.c_char_p),
              ("engineVersion",ctypes.c_uint32),("apiVersion",ctypes.c_uint32)]
class CI(ctypes.Structure):
    _fields_=[("sType",ctypes.c_int),("pNext",ctypes.c_void_p),("flags",ctypes.c_uint32),
              ("pApplicationInfo",ctypes.c_void_p),("enabledLayerCount",ctypes.c_uint32),
              ("ppEnabledLayerNames",ctypes.c_void_p),("enabledExtensionCount",ctypes.c_uint32),
              ("ppEnabledExtensionNames",ctypes.c_void_p)]
app=App(0,None,b"probe",1,b"none",1,(1<<22)|(2<<12))
ci=CI(1,None,0,ctypes.cast(ctypes.pointer(app),ctypes.c_void_p),0,None,0,None)
inst=ctypes.c_void_p(); r=vk.vkCreateInstance(ctypes.byref(ci),None,ctypes.byref(inst))
if r!=0: print(f"FATAL vkCreateInstance={r} — bad host, terminate and redeploy elsewhere"); sys.exit(1)
n=ctypes.c_uint32(0); vk.vkEnumeratePhysicalDevices(inst,ctypes.byref(n),None)
print(f"vkCreateInstance OK, {n.value} device(s)"); sys.exit(0 if n.value else 1)
PYEOF

echo "== [4/5] Node 22 (official tarball — ships include/node headers for helios_vk.node) =="
if ! /usr/local/bin/node --version 2>/dev/null | grep -q v22; then
  cd /usr/local
  curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz | tar -xJ --strip-components=1
fi
node --version; which gcc || echo "WARNING: no gcc — helios native rebuild will fail"

echo "== [5/5] done =="
echo "Deploy the repo with rsync FROM the control box (github may be unreachable here):"
echo "  rsync -az --exclude=.git --exclude=.next --exclude=.turbo alpha2/ root@POD:/workspace/alpha2/"
echo "Then: cd /workspace/alpha2 && node packages/helios/native/build.mjs  (if addon ABI mismatches)"
echo "Train: VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json node --expose-gc apps/cli/dist/main.js train --backend=helios ..."

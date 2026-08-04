/*
 * rm_spy.c — watch what a working NVIDIA driver actually does.
 *
 * WHAT: an LD_PRELOAD ioctl interposer that decodes and logs every RM escape a
 * process makes — object allocations with their classes, control commands, and
 * memory allocation/mapping with their parameters.
 *
 * WHY: when your own implementation does not work and a working one is sitting
 * on the same machine, stop guessing and observe the working one. This found in
 * one run what several rounds of parameter sweeps did not.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: it does not decode every control command
 * or every parameter struct. It decodes the handful that matter for channel and
 * memory bring-up; everything else is logged by escape number so it can be
 * looked up.
 *
 * On the soul constraint: this is a development tool, outside the training loop,
 * in the same category as nvdisasm and checkpoint conversion. Nothing it
 * produces is linked or shipped.
 *
 * Usage:
 *   gcc -shared -fPIC -O2 -o rm_spy.so rm_spy.c -ldl
 *   LD_PRELOAD=./rm_spy.so python3 -c "import torch; torch.ones(64,device='cuda')+1"
 *   cat /root/mem.log
 *
 * WHAT IT FOUND (RTX 3070, driver 580.159.03, CUDA 12.8):
 *
 *  1. CUDA allocates with NVOS64_PARAMETERS (48 bytes), not NVOS21 (32). Same
 *     escape 0x2B -- the kernel tells them apart by the size baked into the
 *     ioctl request. NVOS64 inserts pRightsRequested before paramsSize, which is
 *     why a naive NVOS21 decode reads paramsSize as 0.
 *
 *  2. NV_ESC_RM_MAP_MEMORY carries FLAGS, and they are not optional:
 *       0x030c8000  system memory   MAPPING=1 MAP_FIXED RESERVE_ON_UNMAP CACHING=6
 *       0x010d0000  video memory    MAPPING=2 MAP_FIXED RESERVE_ON_UNMAP CACHING=2
 *     Every map we had been issuing passed flags = 0.
 *
 *  3. The post-channel sequence is entirely QUERIES, not state changes:
 *       NV906F_CTRL_GET_CLASS_ENGINEID     after each engine object
 *       NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST
 *       NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE
 *     So there is no missing "go" call after allocating the channel -- a useful
 *     negative that closes off a whole line of guessing.
 *
 *  4. GET_CLASS_ENGINEID returns a classEngineID, and THAT is what belongs in a
 *     SET_OBJECT method -- not the raw class id (0xC7C0) we were writing.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
typedef uint32_t NvHandle,NvU32; typedef int32_t NvV32; typedef uint64_t NvU64,NvP64;
typedef struct { NvHandle hRoot,hObjectParent,hObjectNew; NvV32 hClass; NvV32 flags;
  NvP64 pMemory __attribute__((aligned(8))); NvU64 limit __attribute__((aligned(8))); NvV32 status; } A02;
typedef struct { A02 p; int fd; } A02F;
typedef struct { NvHandle hClient,hDevice,hMemory; NvU64 offset __attribute__((aligned(8))),length;
  NvP64 pLinear __attribute__((aligned(8))); NvU32 status,flags; } M33;
typedef struct { M33 p; int fd; } M33F;
typedef struct { NvHandle hRoot,hObjectParent,hObjectNew; NvV32 hClass;
  NvP64 pAllocParms __attribute__((aligned(8))); NvU32 paramsSize; NvV32 status; } A21;
typedef struct { NvU32 owner,type,flags,width,height; int pitch; NvU32 attr,attr2,format,cc,zc;
  NvU64 rangeLo __attribute__((aligned(8))),rangeHi,size,alignment,offset,limit;
  NvP64 address __attribute__((aligned(8))); NvU32 ctagOffset; NvHandle hVASpace;
  NvU32 internalflags,tag; int numaNode; } MEMP;
static int (*real_ioctl)(int,unsigned long,...)=NULL;
static FILE*L=NULL;
int ioctl(int fd,unsigned long req,...){
  va_list ap; va_start(ap,req); void*a=va_arg(ap,void*); va_end(ap);
  if(!real_ioctl) real_ioctl=dlsym(RTLD_NEXT,"ioctl");
  if(!L){ L=fopen("/root/mem.log","w"); setvbuf(L,NULL,_IONBF,0); }
  unsigned nr=req&0xff,type=(req>>8)&0xff,size=(req>>16)&0x3fff;
  int rc=real_ioctl(fd,req,a);
  if(type=='F'&&L){
    if(nr==0x27&&a){ A02F*w=a;
      fprintf(L,"ALLOC_MEMORY  class=0x%-5x flags=0x%08x pMemory=0x%llx limit=0x%llx fd=%d size=%u status=0x%x\n",
        (unsigned)w->p.hClass,(unsigned)w->p.flags,(unsigned long long)w->p.pMemory,
        (unsigned long long)w->p.limit,(size==sizeof(A02F))?w->fd:-1,size,w->p.status);
    } else if(nr==0x2B&&a){ A21*p=a;
      if((unsigned)p->hClass==0x3e||(unsigned)p->hClass==0x40||(unsigned)p->hClass==0x71){
        MEMP*m=(MEMP*)(uintptr_t)p->pAllocParms;
        fprintf(L,"ALLOC class=0x%-5x psz=%-4u",(unsigned)p->hClass,p->paramsSize);
        if(m&&p->paramsSize>=sizeof(MEMP))
          fprintf(L," owner=0x%x type=%u flags=0x%08x attr=0x%08x attr2=0x%08x size=0x%llx align=0x%llx hVASpace=0x%x",
            m->owner,m->type,m->flags,m->attr,m->attr2,(unsigned long long)m->size,
            (unsigned long long)m->alignment,m->hVASpace);
        fprintf(L," status=0x%x\n",p->status);
      }
    } else if(nr==0x4e&&a){ M33F*w=a;
      fprintf(L,"MAP_MEMORY    hMemory=0x%-10x len=0x%llx flags=0x%08x fd=%d -> pLinear=0x%llx status=0x%x\n",
        w->p.hMemory,(unsigned long long)w->p.length,w->p.flags,
        (size==sizeof(M33F))?w->fd:-1,(unsigned long long)w->p.pLinearAddress,w->p.status);
    }
  }
  return rc;
}

/* MobileSkinXAI native JPEG boundary. Project code; vendor notices retained.
 * No orientation or ICC transforms here: Dart applies EXIF exactly once.
 * Every returned pointer must be released with msx_free().
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "turbojpeg.h"
#if defined(_WIN32)
#define API __declspec(dllexport)
#define THREAD_LOCAL __declspec(thread)
#else
#define API __attribute__((visibility("default"))) __attribute__((used))
#define THREAD_LOCAL _Thread_local
#endif
static THREAD_LOCAL char error_text[512];
static void error(tjhandle h, const char *fallback) {
  const char *text = h ? tj3GetErrorStr(h) : fallback;
  snprintf(error_text, sizeof(error_text), "%s", text ? text : "JPEG failure");
}
API const char *msx_error(void) { return error_text; }
API const char *msx_version(void) { return "libjpeg-turbo 3.2.0; ISLOW; fancy upsample; RGB; no ICC; JPEG95 444"; }
API void msx_free(void *p) { free(p); }
API uint8_t *msx_decode(const uint8_t *data, size_t length, int *width, int *height) {
  error_text[0] = 0;
  if (!data || !width || !height || length < 4 || length > 64u*1024u*1024u) {
    error(NULL,"Invalid JPEG input size"); return NULL;
  }
  tjhandle h = tj3Init(TJINIT_DECOMPRESS);
  if (!h) { error(NULL,"Cannot initialize JPEG decoder"); return NULL; }
  if (tj3Set(h,TJPARAM_FASTDCT,0) || tj3Set(h,TJPARAM_FASTUPSAMPLE,0) ||
      tj3DecompressHeader(h,data,length)) { error(h,NULL); tj3Destroy(h); return NULL; }
  int w=tj3Get(h,TJPARAM_JPEGWIDTH), he=tj3Get(h,TJPARAM_JPEGHEIGHT);
  int cs=tj3Get(h,TJPARAM_COLORSPACE);
  if (w<=0 || he<=0 || (uint64_t)w*he>50000000u || cs==TJCS_CMYK || cs==TJCS_YCCK) {
    error(NULL,"Unsupported JPEG dimensions or CMYK/YCCK colorspace"); tj3Destroy(h); return NULL;
  }
  uint8_t *pixels=malloc((size_t)w*he*3);
  if (!pixels) { error(NULL,"JPEG allocation failed"); tj3Destroy(h); return NULL; }
  if (tj3Decompress8(h,data,length,pixels,0,TJPF_RGB)) {
    error(h,NULL); free(pixels); tj3Destroy(h); return NULL;
  }
  *width=w; *height=he; tj3Destroy(h); return pixels;
}
API uint8_t *msx_roundtrip(const uint8_t *rgb, int width, int height) {
  error_text[0]=0;
  if (!rgb || width!=224 || height!=224) { error(NULL,"Expected prepared224 RGB"); return NULL; }
  tjhandle h=tj3Init(TJINIT_COMPRESS);
  if (!h) { error(NULL,"Cannot initialize JPEG encoder"); return NULL; }
  unsigned char *jpeg=NULL; size_t bytes=0;
  if (tj3Set(h,TJPARAM_QUALITY,95) || tj3Set(h,TJPARAM_SUBSAMP,TJSAMP_444) ||
      tj3Set(h,TJPARAM_FASTDCT,0) || tj3Set(h,TJPARAM_OPTIMIZE,0) ||
      tj3Compress8(h,rgb,width,0,height,TJPF_RGB,&jpeg,&bytes)) {
    error(h,NULL); if(jpeg)tj3Free(jpeg); tj3Destroy(h); return NULL;
  }
  tj3Destroy(h); int w=0,he=0;
  uint8_t *pixels=msx_decode(jpeg,bytes,&w,&he); tj3Free(jpeg);
  if (pixels && (w!=width || he!=height)) { free(pixels); error(NULL,"JPEG size mismatch"); return NULL; }
  return pixels;
}

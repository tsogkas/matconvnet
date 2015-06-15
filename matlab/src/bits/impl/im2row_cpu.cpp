// @file im2row_cpu.cpp
// @brief Stack image patches as matrix rows (CPU)
// @author Andrea Vedaldi

/*
Copyright (C) 20114-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2row.hpp"
#include <string.h>

using namespace vl ;
using namespace vl::impl ;

/* ---------------------------------------------------------------- */
/*                                                  Heper functions */
/* ---------------------------------------------------------------- */

static inline int floor_divide(int a, int b) {
  if (a >= 0) return a/b;
  else return (a - b + 1)/b;
}

static inline int ceil_divide(int a, int b) {
  if (a >= 0) return (a + b - 1)/b ;
  else return a/b ;
}

static inline int static_max(int a, int b) {
  return (a>=b) ? a:b ;
}

static inline int static_min(int a, int b) {
  return (a<=b) ? a:b ;
}

/* ---------------------------------------------------------------- */
/*                                                           im2row */
/* ---------------------------------------------------------------- */

/* TODO: must transpose */

template <typename T> static inline void
im2row_cpu(T* stacked,          // #pixelsInPatch x #nPatches
           T const* data,
           size_t width,        // input width
           size_t height,       // input height
           size_t depth,
           size_t windowWidth,  // kernel width
           size_t windowHeight, // kernel height
           size_t strideX,
           size_t strideY,
           size_t padLeft,
           size_t padRight,
           size_t padTop,
           size_t padBottom,
           size_t holeX,        // STATSOGK (hole_w in gpapan code)
           size_t holeY)        // STATSOGK (hole_h in gpapan code)
{
  int windowHeightEff = windowHeight + (windowHeight-1) * (holeY - 1);  //(kernel_h_eff in gpapan code)
  int windowWidthEff  = windowWidth  + (windowWidth-1)  * (holeX - 1);  //(kernel_w_eff in gpapan code)
  int numPatchesX = (width + (padLeft + padRight) - windowWidthEff)/strideX + 1 ;   // width_col in gpapan code
  int numPatchesY = (height + (padTop + padBottom) - windowHeightEff)/strideY + 1 ; // height_col in gpapan code
  int numRows = windowWidth * windowHeight * depth ; // # channels_col in gpapan code elements within each patch (from all 3 dimensions)

  /*
   Fill a row of the stacked image at a time. Since patches are stored
   along the columns, scanning a row menas visiting all patche once.
   Each row corresponds to a particular offset within each patch.

   In this manner, as we fill a row
   we tend to access spatially adiacent elements
   in the input image, particulary for small strides.
   */
  for (int row = 0; row < numRows ; ++row) {  // row is c in gpapan code
    /*
     Get the patch offset corresponding to this row of the stacked
     image.
     */
    int u = row ;
    int v = u / windowWidth ;
    int z = v / windowHeight ;         // c_im in gpapan code
    u  = (u % windowWidth)  * holeX ;  // w_offset in gpapan code
    v  = (v % windowHeight) * holeY ;  // h_offset in gpapan code

    /*
     Filling this row amounts to visiting all the pixels in the input
     image that appear at a given offset in the outut patches. Accounting
     for the subsampling of the output patches and input padding,
     these pixels are given by

     x_data(x) = x * strideX + u - padLeft,  0 <= x < numPatchesX
     y_data(y) = y * strideY + v - padTop,   0 <= y < numPatchesY
     z_data(z) = z.

     Here (x,y) are the spatial indexes of the output patches. Depending
     on the padding, some of these values will read pixels outside
     the input image, which should default to 0. In particular, x lands
     on a x_data(x) within the image if x0 <= x < x1 where:

     x_data(x) >= 0 <=> x >= (padLeft - u) / stride
                    <=> x >= ceil((padLeft - u) / stride) = x0
     x_data(x) <= width-1 <=> x <= (width-1 + padLeft - u) / stride
                          <=> x <= floor((width-1 + padLeft - u) / stride)
                          <=> x <  floor((width-1 + padLeft - u) / stride) + 1 = x1

     and the same for y. Note that, while usually x0 <= x1, there are
     special cases for which x1 < x0. This is accounted for in the loops
     below.
     */

    int x0 = static_min(numPatchesX, ceil_divide(padLeft - u, strideX)) ;
    int y0 = static_min(numPatchesY, ceil_divide(padTop - v, strideY)) ;
    int x1 = static_min(numPatchesX, floor_divide(width-1 + padLeft - u, strideX) + 1) ;
    int y1 = static_min(numPatchesY, floor_divide(height-1 + padTop - v, strideY) + 1) ;
    int x ;
    int y ;


    for (y = 0 ; y < y0 ; ++y) {        // zero-pad top side of output
      for (x = 0 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }
    for ( ; y < y1 ; ++y) {
      for (x = 0 ; x < x0 ; ++x) {      // zero-pad left side of output
        *stacked++ = 0 ;
      }
      int y_data = y * strideY + v - padTop ;   // h_im in gpapan code
      int x_data = x * strideX + u - padLeft ;  // w_im in gpapan code
      T const * b = data + (z * height + y_data) * width + x_data ;
      for ( ; x < x1 ; ++x) { // assign input value to corresponding output pixel for all patches
        *stacked++ = *b ;
        b += strideX ;
      }
      for ( ; x < numPatchesX ; ++x) {  // zero-pad right side
        *stacked++ = 0 ;
      }
    }
    for ( ; y < numPatchesY ; ++y) {    // zero-pad bottom side
      for (x = 0 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }
  }
}

template <> vl::Error
vl::impl::im2row<vl::CPU, float>(vl::Context& context,
                                 float* stacked,
                                 float const* data,
                                 size_t height, size_t width, size_t depth,
                                 size_t windowHeight, size_t windowWidth,
                                 size_t strideY, size_t strideX,
                                 size_t padTop, size_t padBottom,
                                 size_t padLeft, size_t padRight,
                                 size_t holeX, size_t holeY)
{
  im2row_cpu<float>(stacked, data,
                    height, width, depth,
                    windowHeight, windowWidth,
                    strideY, strideX,
                    padTop, padBottom, padLeft, padRight,holeX,holeY) ;
  return vlSuccess ;
}

/* ---------------------------------------------------------------- */
/*                                                           row2im */
/* ---------------------------------------------------------------- */

/* TODO: must transpose */

template <typename T> static inline void
row2im_cpu(T* data,
           T const* stacked,
           size_t width,
           size_t height,
           size_t depth,
           size_t windowWidth,
           size_t windowHeight,
           size_t strideX,
           size_t strideY,
           size_t padLeft,
           size_t padRight,
           size_t padTop,
           size_t padBottom,
           size_t holeX,
           size_t holeY)
{
  int windowHeightEff = windowHeight + (windowHeight-1) * (holeY - 1);  //(kernel_h_eff in gpapan code)
  int windowWidthEff  = windowWidth  + (windowWidth-1)  * (holeX - 1);  //(kernel_w_eff in gpapan code)
  int numPatchesX = (width + (padLeft + padRight) - windowWidthEff)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeightEff)/strideY + 1 ;
  int numRows = windowWidth * windowHeight * depth ;

  memset(data, 0, sizeof(T) * width * height * depth) ;

  /*
   Do the converse of im2col, still scanning rows of the stacked image.
   See comments of im2col for an explanation of the algorithm.
   */
  for (int row = 0; row < numRows ; ++row) {
    int u = row ;
    int v = u / windowWidth ;
    int z = v / windowHeight ;         // c_im in gpapan code
    u  = (u % windowWidth)  * holeX ;  // w_offset in gpapan code
    v  = (v % windowHeight) * holeY ;  // h_offset in gpapan code

    int x0 = static_min(numPatchesX, ceil_divide(padLeft - u, strideX)) ;
    int y0 = static_min(numPatchesY, ceil_divide(padTop - v, strideY)) ;
    int x1 = static_min(numPatchesX, floor_divide(width-1 + padLeft - u, strideX) + 1) ;
    int y1 = static_min(numPatchesY, floor_divide(height-1 + padTop - v, strideY) + 1) ;
    int x ;
    int y ;

    y = static_max(0, y0) ;
    stacked += numPatchesX * static_max(y, 0) ;
    for ( ; y < y1 ; ++y) {
      x = static_max(0, x0) ;
      int y_data = y * strideY + v - padTop ;
      int x_data = x * strideX + u - padLeft ;
      T * b = data + (z * height + y_data) * width + x_data ;
      stacked += x ;
      for ( ; x < x1 ; ++x) {
        *b += *stacked++ ;
        b += strideX ;
      }
      stacked += numPatchesX - x ;
    }
    stacked += numPatchesX * (numPatchesY - y) ;
  }
}

template <> vl::Error
vl::impl::row2im<vl::CPU, float>(vl::Context& context,
                                 float* data,
                                 float const* stacked,
                                 size_t height, size_t width, size_t depth,
                                 size_t windowHeight, size_t windowWidth,
                                 size_t strideY, size_t strideX,
                                 size_t padTop, size_t padBottom,
                                 size_t padLeft, size_t padRight,
                                 size_t holeX, size_t holeY)
{
  row2im_cpu<float>(data, stacked,
                    height, width, depth,
                    windowHeight, windowWidth,
                    strideY, strideX,
                    padTop, padBottom, padLeft, padRight,holeX,holeY) ;
  return vlSuccess ;
}

/*!
 * \file mish_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _MISH_TILING_DATA_H_
#define _MISH_TILING_DATA_H_

struct MishTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的元素数量（512B granule 均匀基长）
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
    int64_t granuleRem = 0;   // 前 r 个核各多 1 个 512B granule（round-5 由 host 直接下发，
                              // kernel 免去从 totalNum 重推 r 的一次 64 位整数除法）
};
#endif

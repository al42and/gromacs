/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

#ifndef GMX_GPU_UTILS_GPU_KERNEL_UTILS_H
#define GMX_GPU_UTILS_GPU_KERNEL_UTILS_H

/*! \internal \file
 *  \brief
 *  NBNXM GPU kernel utility methods
 *
 *  \ingroup module_gpu_utils
 */

#include "config.h"

#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/nbnxm/nbnxm.h"

#include "gputraits.h"

#if GMX_GPU_SYCL
#    define GMX_DEVICE_ATTRIBUTE
#    include "sycl_kernel_utils.h"
#elif GMX_GPU_HIP || GMX_GPU_CUDA
#    define GMX_DEVICE_ATTRIBUTE __device__
#    if GMX_GPU_HIP
#        include "hip_kernel_utils.h"
#    else
#        include "cuda_kernel_utils.cuh"
#    endif
#else
#    error Including shared gpu kernel utilities header in unsupported build config
#endif

#define GMX_ALWAYS_INLINE GMX_DEVICE_ATTRIBUTE __attribute__((always_inline))

class Float2Wrapper
{
public:
    GMX_DEVICE_ATTRIBUTE Float2Wrapper(Float2 input) : storage_(input) {}

    template<typename Index>
    GMX_ALWAYS_INLINE float operator[](Index i) const
    {
        switch (i)
        {
#if GMX_GPU_SYCL
            case 0: return storage_[0];
            default: return storage_[1];
#else
            case 0: return storage_.x;
            default: return storage_.y;
#endif
        }
    }

private:
    Float2 storage_;
};

static inline GMX_ALWAYS_INLINE float gmxGpuFDim(const float one, const float two)
{
#if GMX_GPU_SYCL
    return sycl::fdim(one, two);
#else
    const float value = one - two;
    return value >= 0.0F ? value : 0.0F;
#endif
}

static inline GMX_ALWAYS_INLINE float gmxGpuExp(const float value)
{
#if GMX_GPU_SYCL
    return sycl::exp(value);
#elif GMX_GPU_CUDA
    return exp(value);
#elif GMX_GPU_HIP
    return __expf(value);
#else
    return __exp(value);
#endif
}

template<typename T>
static inline GMX_ALWAYS_INLINE float gmxGpuFma(const T valueOne, const T valueTwo, const T valueThree)
{
#if GMX_GPU_SYCL
    return sycl::fma(valueOne, valueTwo, valueThree);
#elif GMX_GPU_CUDA
    return fma(valueOne, valueTwo, valueThree);
#elif GMX_GPU_HIP
    return __fmaf_rn(valueOne, valueTwo, valueThree);
#else
    return __fma(valueOne, valueTwo, valueThree);
#endif
}

/*! \brief Linear interpolation using exactly two FMA operations.
 *
 *  Implements numeric equivalent of: (1-t)*d0 + t*d1.
 */
template<typename T>
static inline GMX_ALWAYS_INLINE T lerp(T d0, T d1, T t)
{
    return gmxGpuFma(t, d1, gmxGpuFma(-t, d0, d0));
}

static inline GMX_ALWAYS_INLINE Float2Wrapper fastLoad(const Float2* input, int offset)
{
#if GMX_GPU_SYCL
    return input[offset];
#else
    return LDG(&input[offset]);
#endif
}

/*!\brief Helper method to generate faster loads.
 *
 * This method helps hipcc (as late as of rocm 6.2.2, hipcc 6.2.41134-65d174c3e and likely later)
 * to generate faster code for loads where 64-bit scalar + 32-bit vector registers are used instead
 * of 64-bit vector versions, saving a few instructions for computing 64-bit vector addresses.
 */
template<typename T>
static GMX_ALWAYS_INLINE const T& amdNbnxmFastLoad(const T* buffer, unsigned int idx, unsigned int offset = 0)
{
    return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(buffer)
                                       + idx * static_cast<unsigned int>(sizeof(T))
                                       + offset * static_cast<unsigned int>(sizeof(T)));
}

/*!\brief Helper method to generate faster atomic operations.
 *
 * This method helps hipcc (as late as of rocm 6.2.2, hipcc 6.2.41134-65d174c3e and likely later)
 * to generate faster code for atomic operations involving 64bit scaler and 32bit vector registers.
 */
template<typename ValueType>
static GMX_ALWAYS_INLINE void
amdFastAtomicAddForce(ValueType* buffer, unsigned int idx, unsigned int component, float value)
{
    atomicAdd(reinterpret_cast<float*>(reinterpret_cast<char*>(buffer)
                                       + (idx * static_cast<unsigned int>(sizeof(ValueType))
                                          + component * static_cast<unsigned int>(sizeof(float)))),
              value);
}

/*! \brief AMD specific helper class to improve data access. */
template<typename ValueType>
class AmdFastBuffer
{
private:
    const ValueType* buffer;

public:
    GMX_ALWAYS_INLINE AmdFastBuffer(const ValueType* buffer) : buffer(buffer) {}
    template<typename IndexType, std::enable_if_t<std::is_integral<IndexType>::value, bool> = true>
    GMX_ALWAYS_INLINE const ValueType& operator[](IndexType idx) const
    {
        return amdNbnxmFastLoad(buffer, idx);
    }
};


#endif

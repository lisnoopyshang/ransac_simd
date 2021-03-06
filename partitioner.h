/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#ifndef _PARTITIONER_H_
#define _PARTITIONER_H_

#ifndef _OPENCL_COMPILER_
#include <iostream>
#endif

#if !defined(_OPENCL_COMPILER_) && defined(OCL_2_0)
#include <atomic>
#endif
#include "stdio.h"
// Partitioner definition -----------------------------------------------------

typedef struct Partitioner {

    int n_tasks;
    int cut;
    int current;
#ifndef _OPENCL_COMPILER_
    int thread_id;
    int n_threads;
#endif


#ifdef OCL_2_0
    // OpenCL 2.0 support for dynamic partitioning
    int strategy;
#ifdef _OPENCL_COMPILER_
    __global atomic_int *worklist;
    __local int *tmp;
#else
    std::atomic_int *worklist;
#endif
#endif

} Partitioner;

// Partitioning strategies
#define STATIC_PARTITIONING 0
#define DYNAMIC_PARTITIONING 1

// Create a partitioner -------------------------------------------------------

inline Partitioner partitioner_create(int n_tasks, float alpha
#ifndef _OPENCL_COMPILER_
    , int thread_id, int n_threads
#endif
#ifdef OCL_2_0
#ifdef _OPENCL_COMPILER_
    , __global atomic_int *worklist
    , __local int *tmp
#else
    , std::atomic_int *worklist
#endif
#endif
    ) {
    Partitioner p;
    p.n_tasks = n_tasks;
#ifndef _OPENCL_COMPILER_
    p.thread_id = thread_id;
    p.n_threads = n_threads;
#endif
    if(alpha >= 0.0 && alpha <= 1.0) {
        p.cut = p.n_tasks * alpha;//cpu
#ifdef OCL_2_0
        p.strategy = STATIC_PARTITIONING;
#endif
    } else {
#ifdef OCL_2_0
        p.strategy = DYNAMIC_PARTITIONING;
        p.worklist = worklist;
#ifdef _OPENCL_COMPILER_
        p.tmp = tmp;
#endif
#endif
    }
    return p;
}

// Partitioner iterators: first() ---------------------------------------------

#ifndef _OPENCL_COMPILER_

inline int cpu_first(Partitioner *p) {
    {
        //p->current = p->thread_id;
		p->current = p->cut + p->thread_id*(p->n_tasks/p->n_threads);//p->current��ǰid��
    }
    return p->current;
}

#else

inline int gpu_first(Partitioner *p) {
#ifdef OCL_2_0
    if(p->strategy == DYNAMIC_PARTITIONING) {
        if(get_local_id(1) == 0 && get_local_id(0) == 0) {
            p->tmp[0] = atomic_fetch_add(p->worklist, 1);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        p->current = p->tmp[0];
    } else
#endif
    {
        //p->current = p->cut + get_group_id(0);
		p->current = get_group_id(0);
    }
    return p->current;
}

#endif

// Partitioner iterators: more() ----------------------------------------------

#ifndef _OPENCL_COMPILER_

inline bool cpu_more(const Partitioner *p) {
#ifdef OCL_2_0
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return (p->current < p->n_tasks);
    } else
#endif
    {
        //return (p->current < p->cut);
		return (p->current < (p->thread_id+1)*(p->n_tasks/p->n_threads));
    }
}

#else

inline bool gpu_more(const Partitioner *p) {
    //return (p->current < p->n_tasks);
	return (p->current < p->cut);
}

#endif

// Partitioner iterators: next() ----------------------------------------------

#ifndef _OPENCL_COMPILER_

inline int cpu_next(Partitioner *p) {
#ifdef OCL_2_0
    if(p->strategy == DYNAMIC_PARTITIONING) {
        p->current = p->worklist->fetch_add(1);
    } else
#endif
    {
		p->current = p->current + 4;
    }
    return p->current;
}

#else

inline int gpu_next(Partitioner *p) {
#ifdef OCL_2_0
    if(p->strategy == DYNAMIC_PARTITIONING) {
        if(get_local_id(1) == 0 && get_local_id(0) == 0) {
            p->tmp[0] = atomic_fetch_add(p->worklist, 1);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        p->current = p->tmp[0];
    } else
#endif
    {
        p->current = p->current + get_num_groups(0);
    }
    return p->current;
}

#endif

#endif


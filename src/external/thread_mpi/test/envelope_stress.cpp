/*
   This source code file is part of thread_mpi.
   Written by Sander Pronk, Erik Lindahl, and possibly others.

   Copyright (c) 2009, Sander Pronk, Erik Lindahl.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
   1) Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   2) Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   3) Neither the name of the copyright holders nor the
   names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY US ''AS IS'' AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL WE BE LIABLE FOR ANY
   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   If you want to redistribute modifications, please consider that
   scientific software is very special. Version control is crucial -
   bugs must be traceable. We will be happy to consider code for
   inclusion in the official distribution, but derived work should not
   be called official thread_mpi. Details are found in the README & COPYING
   files.
 */

/*
 * Reproducer for the tMPI ev_outgoing_received race condition on AArch64.
 *
 * Background
 * ----------
 * tMPI_Wait_process_incoming() uses tMPI_Atomic_get() — a plain volatile
 * load — to read ev_outgoing_received.  On AArch64 (weak memory model) this
 * bare ldr can observe a value written by a concurrent tMPI_Xfer *after*
 * tMPI_Event_wait() captured check_id, because no acquire barrier separates
 * the event wait from the load.  The inflated value makes
 *   check_id -= n_handled  <= 0
 * which skips the head_new scan that would match and free recv envelopes.
 * Repeated over many rounds under oversubscription the recv-envelope pool
 * (size = (N+1)*N_EV_ALLOC = 3*16 = 48 for N=2) is exhausted and tMPI
 * aborts with:
 *   "Out of receive envelopes: this shouldn't happen (probably a bug)"
 *
 * How this test triggers the race
 * --------------------------------
 *  - N tMPI threads oversubscribe the CPU (request more threads than cores),
 *    causing frequent OS preemption in the race window.
 *  - Every iteration each rank posts IRecv from ALL other ranks, then ISend
 *    to ALL other ranks, then Waitall.  This creates the maximum number of
 *    concurrent tMPI_Xfer calls racing with tMPI_Wait_process_incoming().
 *  - IRecvs are posted first so that recv envelopes are allocated from the
 *    pool before their matching sends arrive, maximising pool pressure.
 *  - Many iterations accumulate unmatched recv envelopes until the pool is
 *    exhausted.
 *
 * Expected outcome
 * ----------------
 *  Without fix : aborts with "Out of receive envelopes" on AArch64 (and
 *                occasionally on oversubscribed x86 too).
 *  With fix    : completes and prints "OK."
 *
 * Usage: envelope_stress -nt <nthreads>
 *   Recommended: nthreads > number of physical CPU cores (e.g. -nt 4 on a
 *   3-core machine, which is the CI configuration that exposed the bug).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_CONFIG_H
#    include "config.h"
#endif

#ifndef MPICC
#    include "tmpi.h"
#else
#    include <mpi.h>
#endif

/* Number of all-to-all rounds.  Enough to exhaust the 48-envelope pool
 * (3*16 for a 2-thread run) many times over if the bug is present. */
#define ENVELOPE_STRESS_ITERS 50000

/* Small message: four ints per message so the transfer is fast and the
 * scheduler spends most of its time in the race window rather than copying. */
#define MSG_INTS 4

static void envelope_stress_tester(const void* /*arg*/)
{
    int myrank, N;

    MPI_Comm_size(MPI_COMM_WORLD, &N);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0)
    {
        printf("Envelope-stress test: %d ranks, %d iterations, %d ints/msg\n",
               N, ENVELOPE_STRESS_ITERS, MSG_INTS);
        fflush(stdout);
    }

    /* Allocate per-peer send/recv buffers and request array. */
    int* send_bufs = (int*)malloc((size_t)(N * MSG_INTS) * sizeof(int));
    int* recv_bufs = (int*)malloc((size_t)(N * MSG_INTS) * sizeof(int));
    /* 2*(N-1) requests: (N-1) recvs + (N-1) sends */
    MPI_Request* reqs = (MPI_Request*)malloc((size_t)(2 * N) * sizeof(MPI_Request));

    if (!send_bufs || !recv_bufs || !reqs)
    {
        fprintf(stderr, "rank %d: malloc failed\n", myrank);
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < ENVELOPE_STRESS_ITERS; iter++)
    {
        /* Fill send buffers with a recognisable pattern. */
        for (int peer = 0; peer < N; peer++)
        {
            int* sb          = send_bufs + peer * MSG_INTS;
            sb[0]            = myrank;
            sb[1]            = peer;
            sb[2]            = iter;
            sb[3]            = myrank ^ peer ^ iter;
        }

        int nreqs = 0;

        /* Post all IRecvs first — this checks out recv envelopes from the
         * pool and maximises the number simultaneously in-flight. */
        for (int peer = 0; peer < N; peer++)
        {
            if (peer == myrank)
            {
                continue;
            }
            if (MPI_Irecv(recv_bufs + peer * MSG_INTS, MSG_INTS, MPI_INT,
                          peer, iter, MPI_COMM_WORLD, &reqs[nreqs++])
                != MPI_SUCCESS)
            {
                fprintf(stderr, "rank %d: MPI_Irecv failed (iter=%d peer=%d)\n",
                        myrank, iter, peer);
                exit(1);
            }
        }

        /* Then post all ISends — each will concurrently run tMPI_Xfer on
         * the destination thread, racing with its tMPI_Wait_process_incoming
         * ev_outgoing_received read. */
        for (int peer = 0; peer < N; peer++)
        {
            if (peer == myrank)
            {
                continue;
            }
            if (MPI_Isend(send_bufs + peer * MSG_INTS, MSG_INTS, MPI_INT,
                          peer, iter, MPI_COMM_WORLD, &reqs[nreqs++])
                != MPI_SUCCESS)
            {
                fprintf(stderr, "rank %d: MPI_Isend failed (iter=%d peer=%d)\n",
                        myrank, iter, peer);
                exit(1);
            }
        }

        if (MPI_Waitall(nreqs, reqs, NULL) != MPI_SUCCESS)
        {
            fprintf(stderr, "rank %d: MPI_Waitall failed (iter=%d)\n",
                    myrank, iter);
            exit(1);
        }

        /* Verify received data. */
        for (int peer = 0; peer < N; peer++)
        {
            if (peer == myrank)
            {
                continue;
            }
            const int* rb = recv_bufs + peer * MSG_INTS;
            if (rb[0] != peer || rb[1] != myrank || rb[2] != iter
                || rb[3] != (peer ^ myrank ^ iter))
            {
                fprintf(stderr,
                        "rank %d: data mismatch from peer %d at iter %d: "
                        "got [%d,%d,%d,%d] expected [%d,%d,%d,%d]\n",
                        myrank, peer, iter, rb[0], rb[1], rb[2], rb[3],
                        peer, myrank, iter, peer ^ myrank ^ iter);
                exit(1);
            }
        }

        if (myrank == 0 && (iter + 1) % 10000 == 0)
        {
            printf("  iter %d/%d\n", iter + 1, ENVELOPE_STRESS_ITERS);
            fflush(stdout);
        }
    }

    free(send_bufs);
    free(recv_bufs);
    free(reqs);

    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0)
    {
        printf("OK.\n");
        fflush(stdout);
    }
}


int main(int argc, char* argv[])
{
    int n;

    if (tMPI_Get_N(&argc, &argv, "-nt", &n) != MPI_SUCCESS)
    {
        fprintf(stderr,
                "envelope_stress: reproducer for the ev_outgoing_received race.\n"
                "Usage: envelope_stress -nt <nthreads>\n"
                "  Use nthreads > physical CPU cores to oversubscribe and\n"
                "  maximise the probability of triggering the ARM race.\n");
        exit(1);
    }

    printf("\nenvelope_stress reproducer.  Number of threads: %d\n\n", n);
    tMPI_Init_fn(1, n, TMPI_AFFINITY_ALL_CORES, envelope_stress_tester, NULL);
    envelope_stress_tester(NULL);
    tMPI_Finalize();
    return 0;
}

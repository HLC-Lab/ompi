/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2009      University of Houston. All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All Rights
 *                         reserved.
 * Copyright (c) 2015-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2018      Siberian State University of Telecommunications
 *                         and Information Science. All rights reserved.
 * Copyright (c) 2022      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c)           Amazon.com, Inc. or its affiliates.
 *                         All rights reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "opal/util/bit_ops.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/op/op.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "coll_base_topo.h"
#include "coll_base_util.h"
#include "coll_base_swing_utils.h"

/*
 * ompi_coll_base_allreduce_intra_nonoverlapping
 *
 * This function just calls a reduce followed by a broadcast
 * both called functions are base but they complete sequentially,
 * i.e. no additional overlapping
 * meaning if the number of segments used is greater than the topo depth
 * then once the first segment of data is fully 'reduced' it is not broadcast
 * while the reduce continues (cost = cost-reduce + cost-bcast + decision x 3)
 *
 */
int
ompi_coll_base_allreduce_intra_nonoverlapping(const void *sbuf, void *rbuf, size_t count,
                                               struct ompi_datatype_t *dtype,
                                               struct ompi_op_t *op,
                                               struct ompi_communicator_t *comm,
                                               mca_coll_base_module_t *module)

{
    int err, rank;

    rank = ompi_comm_rank(comm);

    // if (rank == 0) {
    //   printf("2: NON OVERLAPPING\n");
    //   fflush(stdout);
    // }

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:allreduce_intra_nonoverlapping rank %d", rank));

    /* Reduce to 0 and broadcast. */

    if (MPI_IN_PLACE == sbuf) {
        if (0 == rank) {
            err = comm->c_coll->coll_reduce (MPI_IN_PLACE, rbuf, count, dtype,
                                            op, 0, comm, comm->c_coll->coll_reduce_module);
        } else {
            err = comm->c_coll->coll_reduce (rbuf, NULL, count, dtype, op, 0,
                                            comm, comm->c_coll->coll_reduce_module);
        }
    } else {
        err = comm->c_coll->coll_reduce (sbuf, rbuf, count, dtype, op, 0,
                                        comm, comm->c_coll->coll_reduce_module);
    }
    if (MPI_SUCCESS != err) {
        return err;
    }

    return comm->c_coll->coll_bcast (rbuf, count, dtype, 0, comm,
                                    comm->c_coll->coll_bcast_module);
}

/*
 *   ompi_coll_base_allreduce_intra_recursivedoubling
 *
 *   Function:       Recursive doubling algorithm for allreduce operation
 *   Accepts:        Same as MPI_Allreduce()
 *   Returns:        MPI_SUCCESS or error code
 *
 *   Description:    Implements recursive doubling algorithm for allreduce.
 *                   Original (non-segmented) implementation is used in MPICH-2
 *                   for small and intermediate size messages.
 *                   The algorithm preserves order of operations so it can
 *                   be used both by commutative and non-commutative operations.
 *
 *         Example on 7 nodes:
 *         Initial state
 *         #      0       1      2       3      4       5      6
 *               [0]     [1]    [2]     [3]    [4]     [5]    [6]
 *         Initial adjustment step for non-power of two nodes.
 *         old rank      1              3              5      6
 *         new rank      0              1              2      3
 *                     [0+1]          [2+3]          [4+5]   [6]
 *         Step 1
 *         old rank      1              3              5      6
 *         new rank      0              1              2      3
 *                     [0+1+]         [0+1+]         [4+5+]  [4+5+]
 *                     [2+3+]         [2+3+]         [6   ]  [6   ]
 *         Step 2
 *         old rank      1              3              5      6
 *         new rank      0              1              2      3
 *                     [0+1+]         [0+1+]         [0+1+]  [0+1+]
 *                     [2+3+]         [2+3+]         [2+3+]  [2+3+]
 *                     [4+5+]         [4+5+]         [4+5+]  [4+5+]
 *                     [6   ]         [6   ]         [6   ]  [6   ]
 *         Final adjustment step for non-power of two nodes
 *         #      0       1      2       3      4       5      6
 *              [0+1+] [0+1+] [0+1+]  [0+1+] [0+1+]  [0+1+] [0+1+]
 *              [2+3+] [2+3+] [2+3+]  [2+3+] [2+3+]  [2+3+] [2+3+]
 *              [4+5+] [4+5+] [4+5+]  [4+5+] [4+5+]  [4+5+] [4+5+]
 *              [6   ] [6   ] [6   ]  [6   ] [6   ]  [6   ] [6   ]
 *
 */
int
ompi_coll_base_allreduce_intra_recursivedoubling(const void *sbuf, void *rbuf,
                                                  size_t count,
                                                  struct ompi_datatype_t *dtype,
                                                  struct ompi_op_t *op,
                                                  struct ompi_communicator_t *comm,
                                                  mca_coll_base_module_t *module)
{
    int ret, line, rank, size, adjsize, remote, distance;
    int newrank, newremote, extra_ranks;
    char *tmpsend = NULL, *tmprecv = NULL, *tmpswap = NULL, *inplacebuf_free = NULL, *inplacebuf;
    ptrdiff_t span, gap = 0;

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    // if (rank == 0) {
    //   printf("3: RECURSIVE DOUBLING\n");
    //   fflush(stdout);
    // }

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:allreduce_intra_recursivedoubling rank %d", rank));

    /* Special case for size == 1 */
    if (1 == size) {
        if (MPI_IN_PLACE != sbuf) {
            ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
            if (ret < 0) { line = __LINE__; goto error_hndl; }
        }
        return MPI_SUCCESS;
    }

    /* Allocate and initialize temporary send buffer */
    span = opal_datatype_span(&dtype->super, count, &gap);
    inplacebuf_free = (char*) malloc(span);
    if (NULL == inplacebuf_free) { ret = -1; line = __LINE__; goto error_hndl; }
    inplacebuf = inplacebuf_free - gap;

    if (MPI_IN_PLACE == sbuf) {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, inplacebuf, (char*)rbuf);
        if (ret < 0) { line = __LINE__; goto error_hndl; }
    } else {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, inplacebuf, (char*)sbuf);
        if (ret < 0) { line = __LINE__; goto error_hndl; }
    }

    tmpsend = (char*) inplacebuf;
    tmprecv = (char*) rbuf;

    /* Determine nearest power of two less than or equal to size */
    adjsize = opal_next_poweroftwo (size);
    adjsize >>= 1;

    /* Handle non-power-of-two case:
       - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
       sets new rank to -1.
       - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
       apply appropriate operation, and set new rank to rank/2
       - Everyone else sets rank to rank - extra_ranks
    */
    extra_ranks = size - adjsize;
    if (rank <  (2 * extra_ranks)) {
        if (0 == (rank % 2)) {
            ret = MCA_PML_CALL(send(tmpsend, count, dtype, (rank + 1),
                                    MCA_COLL_BASE_TAG_ALLREDUCE,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
            newrank = -1;
        } else {
            ret = MCA_PML_CALL(recv(tmprecv, count, dtype, (rank - 1),
                                    MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                    MPI_STATUS_IGNORE));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
            /* tmpsend = tmprecv (op) tmpsend */
            ompi_op_reduce(op, tmprecv, tmpsend, count, dtype);
            newrank = rank >> 1;
        }
    } else {
        newrank = rank - extra_ranks;
    }

    /* Communication/Computation loop
       - Exchange message with remote node.
       - Perform appropriate operation taking in account order of operations:
       result = value (op) result
    */
    for (distance = 0x1; distance < adjsize; distance <<=1) {
        if (newrank < 0) break;
        /* Determine remote node */
        newremote = newrank ^ distance;
        remote = (newremote < extra_ranks)?
            (newremote * 2 + 1):(newremote + extra_ranks);

        /* Exchange the data */
        ret = ompi_coll_base_sendrecv_actual(tmpsend, count, dtype, remote,
                                             MCA_COLL_BASE_TAG_ALLREDUCE,
                                             tmprecv, count, dtype, remote,
                                             MCA_COLL_BASE_TAG_ALLREDUCE,
                                             comm, MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        /* Apply operation */
        if (rank < remote) {
            /* tmprecv = tmpsend (op) tmprecv */
            ompi_op_reduce(op, tmpsend, tmprecv, count, dtype);
            tmpswap = tmprecv;
            tmprecv = tmpsend;
            tmpsend = tmpswap;
        } else {
            /* tmpsend = tmprecv (op) tmpsend */
            ompi_op_reduce(op, tmprecv, tmpsend, count, dtype);
        }
    }

    /* Handle non-power-of-two case:
       - Odd ranks less than 2 * extra_ranks send result from tmpsend to
       (rank - 1)
       - Even ranks less than 2 * extra_ranks receive result from (rank + 1)
    */
    if (rank < (2 * extra_ranks)) {
        if (0 == (rank % 2)) {
            ret = MCA_PML_CALL(recv(rbuf, count, dtype, (rank + 1),
                                    MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                    MPI_STATUS_IGNORE));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
            tmpsend = (char*)rbuf;
        } else {
            ret = MCA_PML_CALL(send(tmpsend, count, dtype, (rank - 1),
                                    MCA_COLL_BASE_TAG_ALLREDUCE,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        }
    }

    /* Ensure that the final result is in rbuf */
    if (tmpsend != rbuf) {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, tmpsend);
        if (ret < 0) { line = __LINE__; goto error_hndl; }
    }

    if (NULL != inplacebuf_free) free(inplacebuf_free);
    return MPI_SUCCESS;

 error_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "%s:%4d\tRank %d Error occurred %d\n",
                 __FILE__, line, rank, ret));
    (void)line;  // silence compiler warning
    if (NULL != inplacebuf_free) free(inplacebuf_free);
    return ret;
}

/*
 *   ompi_coll_base_allreduce_intra_ring
 *
 *   Function:       Ring algorithm for allreduce operation
 *   Accepts:        Same as MPI_Allreduce()
 *   Returns:        MPI_SUCCESS or error code
 *
 *   Description:    Implements ring algorithm for allreduce: the message is
 *                   automatically segmented to segment of size M/N.
 *                   Algorithm requires 2*N - 1 steps.
 *
 *   Limitations:    The algorithm DOES NOT preserve order of operations so it
 *                   can be used only for commutative operations.
 *                   In addition, algorithm cannot work if the total count is
 *                   less than size.
 *         Example on 5 nodes:
 *         Initial state
 *   #      0              1             2              3             4
 *        [00]           [10]          [20]           [30]           [40]
 *        [01]           [11]          [21]           [31]           [41]
 *        [02]           [12]          [22]           [32]           [42]
 *        [03]           [13]          [23]           [33]           [43]
 *        [04]           [14]          [24]           [34]           [44]
 *
 *        COMPUTATION PHASE
 *         Step 0: rank r sends block r to rank (r+1) and receives bloc (r-1)
 *                 from rank (r-1) [with wraparound].
 *    #     0              1             2              3             4
 *        [00]          [00+10]        [20]           [30]           [40]
 *        [01]           [11]         [11+21]         [31]           [41]
 *        [02]           [12]          [22]          [22+32]         [42]
 *        [03]           [13]          [23]           [33]         [33+43]
 *      [44+04]          [14]          [24]           [34]           [44]
 *
 *         Step 1: rank r sends block (r-1) to rank (r+1) and receives bloc
 *                 (r-2) from rank (r-1) [with wraparound].
 *    #      0              1             2              3             4
 *         [00]          [00+10]     [01+10+20]        [30]           [40]
 *         [01]           [11]         [11+21]      [11+21+31]        [41]
 *         [02]           [12]          [22]          [22+32]      [22+32+42]
 *      [33+43+03]        [13]          [23]           [33]         [33+43]
 *        [44+04]       [44+04+14]       [24]           [34]           [44]
 *
 *         Step 2: rank r sends block (r-2) to rank (r+1) and receives bloc
 *                 (r-2) from rank (r-1) [with wraparound].
 *    #      0              1             2              3             4
 *         [00]          [00+10]     [01+10+20]    [01+10+20+30]      [40]
 *         [01]           [11]         [11+21]      [11+21+31]    [11+21+31+41]
 *     [22+32+42+02]      [12]          [22]          [22+32]      [22+32+42]
 *      [33+43+03]    [33+43+03+13]     [23]           [33]         [33+43]
 *        [44+04]       [44+04+14]  [44+04+14+24]      [34]           [44]
 *
 *         Step 3: rank r sends block (r-3) to rank (r+1) and receives bloc
 *                 (r-3) from rank (r-1) [with wraparound].
 *    #      0              1             2              3             4
 *         [00]          [00+10]     [01+10+20]    [01+10+20+30]     [FULL]
 *        [FULL]           [11]        [11+21]      [11+21+31]    [11+21+31+41]
 *     [22+32+42+02]     [FULL]          [22]         [22+32]      [22+32+42]
 *      [33+43+03]    [33+43+03+13]     [FULL]          [33]         [33+43]
 *        [44+04]       [44+04+14]  [44+04+14+24]      [FULL]         [44]
 *
 *        DISTRIBUTION PHASE: ring ALLGATHER with ranks shifted by 1.
 *
 */
int
ompi_coll_base_allreduce_intra_ring(const void *sbuf, void *rbuf, size_t count,
                                     struct ompi_datatype_t *dtype,
                                     struct ompi_op_t *op,
                                     struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module)
{
  int ret, line, rank, size, k, recv_from, send_to, block_count, inbi;
    int early_segcount, late_segcount, split_rank, max_segcount;
    size_t typelng;
    char *tmpsend = NULL, *tmprecv = NULL, *inbuf[2] = {NULL, NULL};
    ptrdiff_t true_lb, true_extent, lb, extent;
    ptrdiff_t block_offset, max_real_segsize;
    ompi_request_t *reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    // if (rank == 0) {
    //   printf("4: RING\n");
    //   fflush(stdout);
    // }

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:allreduce_intra_ring rank %d, count %zu", rank, count));

    /* Special case for size == 1 */
    if (1 == size) {
        if (MPI_IN_PLACE != sbuf) {
            ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
            if (ret < 0) { line = __LINE__; goto error_hndl; }
        }
        return MPI_SUCCESS;
    }

    /* Special case for count less than size - use recursive doubling */
    if (count < (size_t) size) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "coll:base:allreduce_ring rank %d/%d, count %zu, switching to recursive doubling", rank, size, count));
        return (ompi_coll_base_allreduce_intra_recursivedoubling(sbuf, rbuf,
                                                                  count,
                                                                  dtype, op,
                                                                  comm, module));
    }

    /* Allocate and initialize temporary buffers */
    ret = ompi_datatype_get_extent(dtype, &lb, &extent);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    ret = ompi_datatype_get_true_extent(dtype, &true_lb, &true_extent);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    ret = ompi_datatype_type_size( dtype, &typelng);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Determine the number of elements per block and corresponding
       block sizes.
       The blocks are divided into "early" and "late" ones:
       blocks 0 .. (split_rank - 1) are "early" and
       blocks (split_rank) .. (size - 1) are "late".
       Early blocks are at most 1 element larger than the late ones.
    */
    COLL_BASE_COMPUTE_BLOCKCOUNT( count, size, split_rank,
                                   early_segcount, late_segcount );
    max_segcount = early_segcount;
    max_real_segsize = true_extent + (max_segcount - 1) * extent;


    inbuf[0] = (char*)malloc(max_real_segsize);
    if (NULL == inbuf[0]) { ret = -1; line = __LINE__; goto error_hndl; }
    if (size > 2) {
        inbuf[1] = (char*)malloc(max_real_segsize);
        if (NULL == inbuf[1]) { ret = -1; line = __LINE__; goto error_hndl; }
    }

    /* Handle MPI_IN_PLACE */
    if (MPI_IN_PLACE != sbuf) {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
        if (ret < 0) { line = __LINE__; goto error_hndl; }
    }

    /* Computation loop */

    /*
       For each of the remote nodes:
       - post irecv for block (r-1)
       - send block (r)
       - in loop for every step k = 2 .. n
       - post irecv for block (r + n - k) % n
       - wait on block (r + n - k + 1) % n to arrive
       - compute on block (r + n - k + 1) % n
       - send block (r + n - k + 1) % n
       - wait on block (r + 1)
       - compute on block (r + 1)
       - send block (r + 1) to rank (r + 1)
       Note that we must be careful when computing the beginning of buffers and
       for send operations and computation we must compute the exact block size.
    */
    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;

    inbi = 0;
    /* Initialize first receive from the neighbor on the left */
    ret = MCA_PML_CALL(irecv(inbuf[inbi], max_segcount, dtype, recv_from,
                             MCA_COLL_BASE_TAG_ALLREDUCE, comm, &reqs[inbi]));
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    /* Send first block (my block) to the neighbor on the right */
    block_offset = ((rank < split_rank)?
                    ((ptrdiff_t)rank * (ptrdiff_t)early_segcount) :
                    ((ptrdiff_t)rank * (ptrdiff_t)late_segcount + split_rank));
    block_count = ((rank < split_rank)? early_segcount : late_segcount);
    tmpsend = ((char*)rbuf) + block_offset * extent;
    ret = MCA_PML_CALL(send(tmpsend, block_count, dtype, send_to,
                            MCA_COLL_BASE_TAG_ALLREDUCE,
                            MCA_PML_BASE_SEND_STANDARD, comm));
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    for (k = 2; k < size; k++) {
        const int prevblock = (rank + size - k + 1) % size;

        inbi = inbi ^ 0x1;

        /* Post irecv for the current block */
        ret = MCA_PML_CALL(irecv(inbuf[inbi], max_segcount, dtype, recv_from,
                                 MCA_COLL_BASE_TAG_ALLREDUCE, comm, &reqs[inbi]));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        /* Wait on previous block to arrive */
        ret = ompi_request_wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        /* Apply operation on previous block: result goes to rbuf
           rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
        */
        block_offset = ((prevblock < split_rank)?
                        ((ptrdiff_t)prevblock * early_segcount) :
                        ((ptrdiff_t)prevblock * late_segcount + split_rank));
        block_count = ((prevblock < split_rank)? early_segcount : late_segcount);
        tmprecv = ((char*)rbuf) + (ptrdiff_t)block_offset * extent;
        ompi_op_reduce(op, inbuf[inbi ^ 0x1], tmprecv, block_count, dtype);

        /* send previous block to send_to */
        ret = MCA_PML_CALL(send(tmprecv, block_count, dtype, send_to,
                                MCA_COLL_BASE_TAG_ALLREDUCE,
                                MCA_PML_BASE_SEND_STANDARD, comm));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    }

    /* Wait on the last block to arrive */
    ret = ompi_request_wait(&reqs[inbi], MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

    /* Apply operation on the last block (from neighbor (rank + 1)
       rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
    recv_from = (rank + 1) % size;
    block_offset = ((recv_from < split_rank)?
                    ((ptrdiff_t)recv_from * early_segcount) :
                    ((ptrdiff_t)recv_from * late_segcount + split_rank));
    block_count = ((recv_from < split_rank)? early_segcount : late_segcount);
    tmprecv = ((char*)rbuf) + (ptrdiff_t)block_offset * extent;
    ompi_op_reduce(op, inbuf[inbi], tmprecv, block_count, dtype);

    /* Distribution loop - variation of ring allgather */
    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;
    for (k = 0; k < size - 1; k++) {
        const int recv_data_from = (rank + size - k) % size;
        const int send_data_from = (rank + 1 + size - k) % size;
        const int send_block_offset =
            ((send_data_from < split_rank)?
             ((ptrdiff_t)send_data_from * early_segcount) :
             ((ptrdiff_t)send_data_from * late_segcount + split_rank));
        const int recv_block_offset =
            ((recv_data_from < split_rank)?
             ((ptrdiff_t)recv_data_from * early_segcount) :
             ((ptrdiff_t)recv_data_from * late_segcount + split_rank));
        block_count = ((send_data_from < split_rank)?
                       early_segcount : late_segcount);

        tmprecv = (char*)rbuf + (ptrdiff_t)recv_block_offset * extent;
        tmpsend = (char*)rbuf + (ptrdiff_t)send_block_offset * extent;

        ret = ompi_coll_base_sendrecv(tmpsend, block_count, dtype, send_to,
                                       MCA_COLL_BASE_TAG_ALLREDUCE,
                                       tmprecv, max_segcount, dtype, recv_from,
                                       MCA_COLL_BASE_TAG_ALLREDUCE,
                                       comm, MPI_STATUS_IGNORE, rank);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl;}

    }

    if (NULL != inbuf[0]) free(inbuf[0]);
    if (NULL != inbuf[1]) free(inbuf[1]);

    return MPI_SUCCESS;

 error_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "%s:%4d\tRank %d Error occurred %d\n",
                 __FILE__, line, rank, ret));
    ompi_coll_base_free_reqs(reqs, 2);
    (void)line;  // silence compiler warning
    if (NULL != inbuf[0]) free(inbuf[0]);
    if (NULL != inbuf[1]) free(inbuf[1]);
    return ret;
}

/*
 *   ompi_coll_base_allreduce_intra_ring_segmented
 *
 *   Function:       Pipelined ring algorithm for allreduce operation
 *   Accepts:        Same as MPI_Allreduce(), segment size
 *   Returns:        MPI_SUCCESS or error code
 *
 *   Description:    Implements pipelined ring algorithm for allreduce:
 *                   user supplies suggested segment size for the pipelining of
 *                   reduce operation.
 *                   The segment size determines the number of phases, np, for
 *                   the algorithm execution.
 *                   The message is automatically divided into blocks of
 *                   approximately  (count / (np * segcount)) elements.
 *                   At the end of reduction phase, allgather like step is
 *                   executed.
 *                   Algorithm requires (np + 1)*(N - 1) steps.
 *
 *   Limitations:    The algorithm DOES NOT preserve order of operations so it
 *                   can be used only for commutative operations.
 *                   In addition, algorithm cannot work if the total size is
 *                   less than size * segment size.
 *         Example on 3 nodes with 2 phases
 *         Initial state
 *   #      0              1             2
 *        [00a]          [10a]         [20a]
 *        [00b]          [10b]         [20b]
 *        [01a]          [11a]         [21a]
 *        [01b]          [11b]         [21b]
 *        [02a]          [12a]         [22a]
 *        [02b]          [12b]         [22b]
 *
 *        COMPUTATION PHASE 0 (a)
 *         Step 0: rank r sends block ra to rank (r+1) and receives bloc (r-1)a
 *                 from rank (r-1) [with wraparound].
 *    #     0              1             2
 *        [00a]        [00a+10a]       [20a]
 *        [00b]          [10b]         [20b]
 *        [01a]          [11a]       [11a+21a]
 *        [01b]          [11b]         [21b]
 *      [22a+02a]        [12a]         [22a]
 *        [02b]          [12b]         [22b]
 *
 *         Step 1: rank r sends block (r-1)a to rank (r+1) and receives bloc
 *                 (r-2)a from rank (r-1) [with wraparound].
 *    #     0              1             2
 *        [00a]        [00a+10a]   [00a+10a+20a]
 *        [00b]          [10b]         [20b]
 *    [11a+21a+01a]      [11a]       [11a+21a]
 *        [01b]          [11b]         [21b]
 *      [22a+02a]    [22a+02a+12a]     [22a]
 *        [02b]          [12b]         [22b]
 *
 *        COMPUTATION PHASE 1 (b)
 *         Step 0: rank r sends block rb to rank (r+1) and receives bloc (r-1)b
 *                 from rank (r-1) [with wraparound].
 *    #     0              1             2
 *        [00a]        [00a+10a]       [20a]
 *        [00b]        [00b+10b]       [20b]
 *        [01a]          [11a]       [11a+21a]
 *        [01b]          [11b]       [11b+21b]
 *      [22a+02a]        [12a]         [22a]
 *      [22b+02b]        [12b]         [22b]
 *
 *         Step 1: rank r sends block (r-1)b to rank (r+1) and receives bloc
 *                 (r-2)b from rank (r-1) [with wraparound].
 *    #     0              1             2
 *        [00a]        [00a+10a]   [00a+10a+20a]
 *        [00b]          [10b]     [0bb+10b+20b]
 *    [11a+21a+01a]      [11a]       [11a+21a]
 *    [11b+21b+01b]      [11b]         [21b]
 *      [22a+02a]    [22a+02a+12a]     [22a]
 *        [02b]      [22b+01b+12b]     [22b]
 *
 *
 *        DISTRIBUTION PHASE: ring ALLGATHER with ranks shifted by 1 (same as
 *         in regular ring algorithm.
 *
 */
int
ompi_coll_base_allreduce_intra_ring_segmented(const void *sbuf, void *rbuf, size_t count,
                                               struct ompi_datatype_t *dtype,
                                               struct ompi_op_t *op,
                                               struct ompi_communicator_t *comm,
                                               mca_coll_base_module_t *module,
                                               uint32_t segsize)
{
    int ret, line, rank, size, k, recv_from, send_to;
    int early_blockcount, late_blockcount, split_rank;
    int num_phases, phase, block_count, inbi;
    size_t typelng, segcount, max_segcount;
    char *tmpsend = NULL, *tmprecv = NULL, *inbuf[2] = {NULL, NULL};
    ptrdiff_t block_offset, max_real_segsize;
    ompi_request_t *reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    ptrdiff_t lb, extent, gap;

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    // if (rank == 0) {
    //   printf("5: RING SEGMENTED\n");
    //   fflush(stdout);
    // }

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:allreduce_intra_ring_segmented rank %d, count %zu", rank, count));

    /* Special case for size == 1 */
    if (1 == size) {
        if (MPI_IN_PLACE != sbuf) {
            ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
            if (ret < 0) { line = __LINE__; goto error_hndl; }
        }
        return MPI_SUCCESS;
    }

    /* Determine segment count based on the suggested segment size */
    ret = ompi_datatype_type_size( dtype, &typelng);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    segcount = count;
    COLL_BASE_COMPUTED_SEGCOUNT(segsize, typelng, segcount)

        /* Special case for count less than size * segcount - use regular ring */
        if (count < (size_t) (size * segcount)) {
            OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "coll:base:allreduce_ring_segmented rank %d/%d, count %zu, switching to regular ring", rank, size, count));
            return (ompi_coll_base_allreduce_intra_ring(sbuf, rbuf, count, dtype, op,
                                                         comm, module));
        }

    /* Determine the number of phases of the algorithm */
    num_phases = count / (size * segcount);
    if ((count % (size * segcount) >= (size_t) size) &&
        (count % (size * segcount) > (size_t) ((size * segcount) / 2))) {
        num_phases++;
    }

    /* Determine the number of elements per block and corresponding
       block sizes.
       The blocks are divided into "early" and "late" ones:
       blocks 0 .. (split_rank - 1) are "early" and
       blocks (split_rank) .. (size - 1) are "late".
       Early blocks are at most 1 element larger than the late ones.
       Note, these blocks will be split into num_phases segments,
       out of the largest one will have max_segcount elements.
    */
    COLL_BASE_COMPUTE_BLOCKCOUNT( count, size, split_rank,
                                   early_blockcount, late_blockcount );
    COLL_BASE_COMPUTE_BLOCKCOUNT( early_blockcount, num_phases, inbi,
                                   max_segcount, k);

    ret = ompi_datatype_get_extent(dtype, &lb, &extent);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
     max_real_segsize = opal_datatype_span(&dtype->super, max_segcount, &gap);

    /* Allocate and initialize temporary buffers */
    inbuf[0] = (char*)malloc(max_real_segsize);
    if (NULL == inbuf[0]) { ret = -1; line = __LINE__; goto error_hndl; }
    if (size > 2) {
        inbuf[1] = (char*)malloc(max_real_segsize);
        if (NULL == inbuf[1]) { ret = -1; line = __LINE__; goto error_hndl; }
    }

    /* Handle MPI_IN_PLACE */
    if (MPI_IN_PLACE != sbuf) {
        ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
        if (ret < 0) { line = __LINE__; goto error_hndl; }
    }

    /* Computation loop: for each phase, repeat ring allreduce computation loop */
    for (phase = 0; phase < num_phases; phase ++) {
        ptrdiff_t phase_offset;
        int early_phase_segcount, late_phase_segcount, split_phase, phase_count;

        /*
           For each of the remote nodes:
           - post irecv for block (r-1)
           - send block (r)
           To do this, first compute block offset and count, and use block offset
           to compute phase offset.
           - in loop for every step k = 2 .. n
           - post irecv for block (r + n - k) % n
           - wait on block (r + n - k + 1) % n to arrive
           - compute on block (r + n - k + 1) % n
           - send block (r + n - k + 1) % n
           - wait on block (r + 1)
           - compute on block (r + 1)
           - send block (r + 1) to rank (r + 1)
           Note that we must be careful when computing the beginning of buffers and
           for send operations and computation we must compute the exact block size.
        */
        send_to = (rank + 1) % size;
        recv_from = (rank + size - 1) % size;

        inbi = 0;
        /* Initialize first receive from the neighbor on the left */
        ret = MCA_PML_CALL(irecv(inbuf[inbi], max_segcount, dtype, recv_from,
                                 MCA_COLL_BASE_TAG_ALLREDUCE, comm, &reqs[inbi]));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        /* Send first block (my block) to the neighbor on the right:
           - compute my block and phase offset
           - send data */
        block_offset = ((rank < split_rank)?
                        ((ptrdiff_t)rank * (ptrdiff_t)early_blockcount) :
                        ((ptrdiff_t)rank * (ptrdiff_t)late_blockcount + split_rank));
        block_count = ((rank < split_rank)? early_blockcount : late_blockcount);
        COLL_BASE_COMPUTE_BLOCKCOUNT(block_count, num_phases, split_phase,
                                      early_phase_segcount, late_phase_segcount)
        phase_count = ((phase < split_phase)?
                       (early_phase_segcount) : (late_phase_segcount));
        phase_offset = ((phase < split_phase)?
                        ((ptrdiff_t)phase * (ptrdiff_t)early_phase_segcount) :
                        ((ptrdiff_t)phase * (ptrdiff_t)late_phase_segcount + split_phase));
        tmpsend = ((char*)rbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
        ret = MCA_PML_CALL(send(tmpsend, phase_count, dtype, send_to,
                                MCA_COLL_BASE_TAG_ALLREDUCE,
                                MCA_PML_BASE_SEND_STANDARD, comm));
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        for (k = 2; k < size; k++) {
            const int prevblock = (rank + size - k + 1) % size;

            inbi = inbi ^ 0x1;

            /* Post irecv for the current block */
            ret = MCA_PML_CALL(irecv(inbuf[inbi], max_segcount, dtype, recv_from,
                                     MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                     &reqs[inbi]));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

            /* Wait on previous block to arrive */
            ret = ompi_request_wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

            /* Apply operation on previous block: result goes to rbuf
               rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
            */
            block_offset = ((prevblock < split_rank)?
                            ((ptrdiff_t)prevblock * (ptrdiff_t)early_blockcount) :
                            ((ptrdiff_t)prevblock * (ptrdiff_t)late_blockcount + split_rank));
            block_count = ((prevblock < split_rank)?
                           early_blockcount : late_blockcount);
            COLL_BASE_COMPUTE_BLOCKCOUNT(block_count, num_phases, split_phase,
                                          early_phase_segcount, late_phase_segcount)
                phase_count = ((phase < split_phase)?
                               (early_phase_segcount) : (late_phase_segcount));
            phase_offset = ((phase < split_phase)?
                            ((ptrdiff_t)phase * (ptrdiff_t)early_phase_segcount) :
                            ((ptrdiff_t)phase * (ptrdiff_t)late_phase_segcount + split_phase));
            tmprecv = ((char*)rbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
            ompi_op_reduce(op, inbuf[inbi ^ 0x1], tmprecv, phase_count, dtype);

            /* send previous block to send_to */
            ret = MCA_PML_CALL(send(tmprecv, phase_count, dtype, send_to,
                                    MCA_COLL_BASE_TAG_ALLREDUCE,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
        }

        /* Wait on the last block to arrive */
        ret = ompi_request_wait(&reqs[inbi], MPI_STATUS_IGNORE);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }

        /* Apply operation on the last block (from neighbor (rank + 1)
           rbuf[rank+1] = inbuf[inbi] (op) rbuf[rank + 1] */
        recv_from = (rank + 1) % size;
        block_offset = ((recv_from < split_rank)?
                        ((ptrdiff_t)recv_from * (ptrdiff_t)early_blockcount) :
                        ((ptrdiff_t)recv_from * (ptrdiff_t)late_blockcount + split_rank));
        block_count = ((recv_from < split_rank)?
                       early_blockcount : late_blockcount);
        COLL_BASE_COMPUTE_BLOCKCOUNT(block_count, num_phases, split_phase,
                                      early_phase_segcount, late_phase_segcount)
            phase_count = ((phase < split_phase)?
                           (early_phase_segcount) : (late_phase_segcount));
        phase_offset = ((phase < split_phase)?
                        ((ptrdiff_t)phase * (ptrdiff_t)early_phase_segcount) :
                        ((ptrdiff_t)phase * (ptrdiff_t)late_phase_segcount + split_phase));
        tmprecv = ((char*)rbuf) + (ptrdiff_t)(block_offset + phase_offset) * extent;
        ompi_op_reduce(op, inbuf[inbi], tmprecv, phase_count, dtype);
    }

    /* Distribution loop - variation of ring allgather */
    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;
    for (k = 0; k < size - 1; k++) {
        const int recv_data_from = (rank + size - k) % size;
        const int send_data_from = (rank + 1 + size - k) % size;
        const int send_block_offset =
            ((send_data_from < split_rank)?
             ((ptrdiff_t)send_data_from * (ptrdiff_t)early_blockcount) :
             ((ptrdiff_t)send_data_from * (ptrdiff_t)late_blockcount + split_rank));
        const int recv_block_offset =
            ((recv_data_from < split_rank)?
             ((ptrdiff_t)recv_data_from * (ptrdiff_t)early_blockcount) :
             ((ptrdiff_t)recv_data_from * (ptrdiff_t)late_blockcount + split_rank));
        block_count = ((send_data_from < split_rank)?
                       early_blockcount : late_blockcount);

        tmprecv = (char*)rbuf + (ptrdiff_t)recv_block_offset * extent;
        tmpsend = (char*)rbuf + (ptrdiff_t)send_block_offset * extent;

        ret = ompi_coll_base_sendrecv(tmpsend, block_count, dtype, send_to,
                                       MCA_COLL_BASE_TAG_ALLREDUCE,
                                       tmprecv, early_blockcount, dtype, recv_from,
                                       MCA_COLL_BASE_TAG_ALLREDUCE,
                                       comm, MPI_STATUS_IGNORE, rank);
        if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl;}

    }

    if (NULL != inbuf[0]) free(inbuf[0]);
    if (NULL != inbuf[1]) free(inbuf[1]);

    return MPI_SUCCESS;

 error_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "%s:%4d\tRank %d Error occurred %d\n",
                 __FILE__, line, rank, ret));
    ompi_coll_base_free_reqs(reqs, 2);
    (void)line;  // silence compiler warning
    if (NULL != inbuf[0]) free(inbuf[0]);
    if (NULL != inbuf[1]) free(inbuf[1]);
    return ret;
}

/*
 * Linear functions are copied from the BASIC coll module
 * they do not segment the message and are simple implementations
 * but for some small number of nodes and/or small data sizes they
 * are just as fast as base/tree based segmenting operations
 * and as such may be selected by the decision functions
 * These are copied into this module due to the way we select modules
 * in V1. i.e. in V2 we will handle this differently and so will not
 * have to duplicate code.
 * GEF Oct05 after asking Jeff.
 */

/* copied function (with appropriate renaming) starts here */


/*
 *	allreduce_intra
 *
 *	Function:	- allreduce using other MPI collectives
 *	Accepts:	- same as MPI_Allreduce()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
ompi_coll_base_allreduce_intra_basic_linear(const void *sbuf, void *rbuf, size_t count,
                                             struct ompi_datatype_t *dtype,
                                             struct ompi_op_t *op,
                                             struct ompi_communicator_t *comm,
                                             mca_coll_base_module_t *module)
{
    int err, rank;

    rank = ompi_comm_rank(comm);

    // if (rank == 0) {
    //   printf("1: LINEAR\n");
    //   fflush(stdout);
    // }
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"coll:base:allreduce_intra_basic_linear rank %d", rank));

    /* Reduce to 0 and broadcast. */

    if (MPI_IN_PLACE == sbuf) {
        if (0 == rank) {
            err = ompi_coll_base_reduce_intra_basic_linear (MPI_IN_PLACE, rbuf, count, dtype,
                                                             op, 0, comm, module);
        } else {
            err = ompi_coll_base_reduce_intra_basic_linear(rbuf, NULL, count, dtype,
                                                            op, 0, comm, module);
        }
    } else {
        err = ompi_coll_base_reduce_intra_basic_linear(sbuf, rbuf, count, dtype,
                                                        op, 0, comm, module);
    }
    if (MPI_SUCCESS != err) {
        return err;
    }

    return ompi_coll_base_bcast_intra_basic_linear(rbuf, count, dtype, 0, comm, module);
}

/*
 * ompi_coll_base_allreduce_intra_redscat_allgather
 *
 * Function:  Allreduce using Rabenseifner's algorithm.
 * Accepts:   Same arguments as MPI_Allreduce
 * Returns:   MPI_SUCCESS or error code
 *
 * Description: an implementation of Rabenseifner's allreduce algorithm [1, 2].
 *   [1] Rajeev Thakur, Rolf Rabenseifner and William Gropp.
 *       Optimization of Collective Communication Operations in MPICH //
 *       The Int. Journal of High Performance Computing Applications. Vol 19,
 *       Issue 1, pp. 49--66.
 *   [2] http://www.hlrs.de/mpi/myreduce.html.
 *
 * This algorithm is a combination of a reduce-scatter implemented with
 * recursive vector halving and recursive distance doubling, followed either
 * by an allgather implemented with recursive doubling [1].
 *
 * Step 1. If the number of processes is not a power of two, reduce it to
 * the nearest lower power of two (p' = 2^{\floor{\log_2 p}})
 * by removing r = p - p' extra processes as follows. In the first 2r processes
 * (ranks 0 to 2r - 1), all the even ranks send the second half of the input
 * vector to their right neighbor (rank + 1), and all the odd ranks send
 * the first half of the input vector to their left neighbor (rank - 1).
 * The even ranks compute the reduction on the first half of the vector and
 * the odd ranks compute the reduction on the second half. The odd ranks then
 * send the result to their left neighbors (the even ranks). As a result,
 * the even ranks among the first 2r processes now contain the reduction with
 * the input vector on their right neighbors (the odd ranks). These odd ranks
 * do not participate in the rest of the algorithm, which leaves behind
 * a power-of-two number of processes. The first r even-ranked processes and
 * the last p - 2r processes are now renumbered from 0 to p' - 1.
 *
 * Step 2. The remaining processes now perform a reduce-scatter by using
 * recursive vector halving and recursive distance doubling. The even-ranked
 * processes send the second half of their buffer to rank + 1 and the odd-ranked
 * processes send the first half of their buffer to rank - 1. All processes
 * then compute the reduction between the local buffer and the received buffer.
 * In the next log_2(p') - 1 steps, the buffers are recursively halved, and the
 * distance is doubled. At the end, each of the p' processes has 1 / p' of the
 * total reduction result.
 *
 * Step 3. An allgather is performed by using recursive vector doubling and
 * distance halving. All exchanges are executed in reverse order relative
 * to recursive doubling on previous step. If the number of processes is not
 * a power of two, the total result vector must be sent to the r processes
 * that were removed in the first step.
 *
 * Limitations:
 *   count >= 2^{\floor{\log_2 p}}
 *   commutative operations only
 *   intra-communicators only
 *
 * Memory requirements (per process):
 *   count * typesize + 4 * \log_2(p) * sizeof(int) = O(count)
 */
int ompi_coll_base_allreduce_intra_redscat_allgather(
    const void *sbuf, void *rbuf, size_t count, struct ompi_datatype_t *dtype,
    struct ompi_op_t *op, struct ompi_communicator_t *comm,
    mca_coll_base_module_t *module)
{
    int *rindex = NULL, *rcount = NULL, *sindex = NULL, *scount = NULL;

    int comm_size = ompi_comm_size(comm);
    int rank = ompi_comm_rank(comm);
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:allreduce_intra_redscat_allgather: rank %d/%d",
                 rank, comm_size));

    // if (rank == 0) {
    //   printf("6: RABENSEIFNER\n");
    //   fflush(stdout);
    // }

    if (!ompi_op_is_commute(op)) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                     "coll:base:allreduce_intra_redscat_allgather: rank %d/%d "
                     "count %zu switching to basic linear allreduce",
                     rank, comm_size, count));
        return ompi_coll_base_allreduce_intra_basic_linear(sbuf, rbuf, count, dtype,
                                                           op, comm, module);
    }

    /* Find nearest power-of-two less than or equal to comm_size */
    int nsteps = opal_hibit(comm_size, comm->c_cube_dim + 1);   /* ilog2(comm_size) */
    if (-1 == nsteps) {
        return MPI_ERR_ARG;
    }
    int nprocs_pof2 = 1 << nsteps;                              /* flp2(comm_size) */
    int err = MPI_SUCCESS;
    ptrdiff_t lb, extent, dsize, gap = 0;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    dsize = opal_datatype_span(&dtype->super, count >> 1, &gap);

    /* Temporary buffer for receiving messages */
    char *tmp_buf = NULL;
    char *tmp_buf_raw = (char *)malloc(dsize);
    if (NULL == tmp_buf_raw)
        return OMPI_ERR_OUT_OF_RESOURCE;
    tmp_buf = tmp_buf_raw - gap;

    if (sbuf != MPI_IN_PLACE) {
        err = ompi_datatype_copy_content_same_ddt(dtype, count, (char *)rbuf,
                                                  (char *)sbuf);
        if (MPI_SUCCESS != err) { goto cleanup_and_return; }
    }

    /*
     * Step 1. Reduce the number of processes to the nearest lower power of two
     * p' = 2^{\floor{\log_2 p}} by removing r = p - p' processes.
     * 1. In the first 2r processes (ranks 0 to 2r - 1), all the even ranks send
     *    the second half of the input vector to their right neighbor (rank + 1)
     *    and all the odd ranks send the first half of the input vector to their
     *    left neighbor (rank - 1).
     * 2. All 2r processes compute the reduction on their half.
     * 3. The odd ranks then send the result to their left neighbors
     *    (the even ranks).
     *
     * The even ranks (0 to 2r - 1) now contain the reduction with the input
     * vector on their right neighbors (the odd ranks). The first r even
     * processes and the p - 2r last processes are renumbered from
     * 0 to 2^{\floor{\log_2 p}} - 1.
     */

    int vrank, step, wsize;
    int nprocs_rem = comm_size - nprocs_pof2;

    if (rank < 2 * nprocs_rem) {
        int count_lhalf = count / 2;
        int count_rhalf = count - count_lhalf;

        if (rank % 2 != 0) {
            /*
             * Odd process -- exchange with rank - 1
             * Send the left half of the input vector to the left neighbor,
             * Recv the right half of the input vector from the left neighbor
             */
            err = ompi_coll_base_sendrecv(rbuf, count_lhalf, dtype, rank - 1,
                                          MCA_COLL_BASE_TAG_ALLREDUCE,
                                          (char *)tmp_buf,
                                          count_rhalf, dtype, rank - 1,
                                          MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                          MPI_STATUS_IGNORE, rank);
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            /* Reduce on the right half of the buffers (result in rbuf) */
            ompi_op_reduce(op, (char *)tmp_buf,
                           (char *)rbuf + count_lhalf * extent, count_rhalf, dtype);

            /* Send the right half to the left neighbor */
            err = MCA_PML_CALL(send((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                    count_rhalf, dtype, rank - 1,
                                    MCA_COLL_BASE_TAG_ALLREDUCE,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            /* This process does not pariticipate in recursive doubling phase */
            vrank = -1;

        } else {
            /*
             * Even process -- exchange with rank + 1
             * Send the right half of the input vector to the right neighbor,
             * Recv the left half of the input vector from the right neighbor
             */
            err = ompi_coll_base_sendrecv((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                          count_rhalf, dtype, rank + 1,
                                          MCA_COLL_BASE_TAG_ALLREDUCE,
                                          tmp_buf, count_lhalf, dtype, rank + 1,
                                          MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                          MPI_STATUS_IGNORE, rank);
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            /* Reduce on the right half of the buffers (result in rbuf) */
            ompi_op_reduce(op, tmp_buf, rbuf, count_lhalf, dtype);

            /* Recv the right half from the right neighbor */
            err = MCA_PML_CALL(recv((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                    count_rhalf, dtype, rank + 1,
                                    MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                    MPI_STATUS_IGNORE));
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            vrank = rank / 2;
        }
    } else { /* rank >= 2 * nprocs_rem */
        vrank = rank - nprocs_rem;
    }

    /*
     * Step 2. Reduce-scatter implemented with recursive vector halving and
     * recursive distance doubling. We have p' = 2^{\floor{\log_2 p}}
     * power-of-two number of processes with new ranks (vrank) and result in rbuf.
     *
     * The even-ranked processes send the right half of their buffer to rank + 1
     * and the odd-ranked processes send the left half of their buffer to
     * rank - 1. All processes then compute the reduction between the local
     * buffer and the received buffer. In the next \log_2(p') - 1 steps, the
     * buffers are recursively halved, and the distance is doubled. At the end,
     * each of the p' processes has 1 / p' of the total reduction result.
     */
    rindex = malloc(sizeof(*rindex) * nsteps);
    sindex = malloc(sizeof(*sindex) * nsteps);
    rcount = malloc(sizeof(*rcount) * nsteps);
    scount = malloc(sizeof(*scount) * nsteps);
    if (NULL == rindex || NULL == sindex || NULL == rcount || NULL == scount) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        goto cleanup_and_return;
    }

    if (vrank != -1) {
        step = 0;
        wsize = count;
        sindex[0] = rindex[0] = 0;

        for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
            /*
             * On each iteration: rindex[step] = sindex[step] -- beginning of the
             * current window. Length of the current window is storded in wsize.
             */
            int vdest = vrank ^ mask;
            /* Translate vdest virtual rank to real rank */
            int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;

            if (rank < dest) {
                /*
                 * Recv into the left half of the current window, send the right
                 * half of the window to the peer (perform reduce on the left
                 * half of the current window)
                 */
                rcount[step] = wsize / 2;
                scount[step] = wsize - rcount[step];
                sindex[step] = rindex[step] + rcount[step];
            } else {
                /*
                 * Recv into the right half of the current window, send the left
                 * half of the window to the peer (perform reduce on the right
                 * half of the current window)
                 */
                scount[step] = wsize / 2;
                rcount[step] = wsize - scount[step];
                rindex[step] = sindex[step] + scount[step];
            }

            /* Send part of data from the rbuf, recv into the tmp_buf */
            err = ompi_coll_base_sendrecv((char *)rbuf + (ptrdiff_t)sindex[step] * extent,
                                          scount[step], dtype, dest,
                                          MCA_COLL_BASE_TAG_ALLREDUCE,
                                          (char *)tmp_buf, rcount[step], dtype, dest,
                                          MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                          MPI_STATUS_IGNORE, rank);
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }

            /* Local reduce: rbuf[] = tmp_buf[] <op> rbuf[] */
            ompi_op_reduce(op, (char *)tmp_buf,
                           (char *)rbuf + (ptrdiff_t)rindex[step] * extent,
                           rcount[step], dtype);

            /* Move the current window to the received message */
            if (step + 1 < nsteps) {
                rindex[step + 1] = rindex[step];
                sindex[step + 1] = rindex[step];
                wsize = rcount[step];
                step++;
            }
        }
        /*
         * Assertion: each process has 1 / p' of the total reduction result:
         * rcount[nsteps - 1] elements in the rbuf[rindex[nsteps - 1], ...].
         */

        /*
         * Step 3. Allgather by the recursive doubling algorithm.
         * Each process has 1 / p' of the total reduction result:
         * rcount[nsteps - 1] elements in the rbuf[rindex[nsteps - 1], ...].
         * All exchanges are executed in reverse order relative
         * to recursive doubling (previous step).
         */

        step = nsteps - 1;

        for (int mask = nprocs_pof2 >> 1; mask > 0; mask >>= 1) {
            int vdest = vrank ^ mask;
            /* Translate vdest virtual rank to real rank */
            int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;

            /*
             * Send rcount[step] elements from rbuf[rindex[step]...]
             * Recv scount[step] elements to rbuf[sindex[step]...]
             */
            err = ompi_coll_base_sendrecv((char *)rbuf + (ptrdiff_t)rindex[step] * extent,
                                          rcount[step], dtype, dest,
                                          MCA_COLL_BASE_TAG_ALLREDUCE,
                                          (char *)rbuf + (ptrdiff_t)sindex[step] * extent,
                                          scount[step], dtype, dest,
                                          MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                          MPI_STATUS_IGNORE, rank);
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }
            step--;
        }
    }

    /*
     * Step 4. Send total result to excluded odd ranks.
     */
    if (rank < 2 * nprocs_rem) {
        if (rank % 2 != 0) {
            /* Odd process -- recv result from rank - 1 */
            err = MCA_PML_CALL(recv(rbuf, count, dtype, rank - 1,
                                    MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                    MPI_STATUS_IGNORE));
            if (OMPI_SUCCESS != err) { goto cleanup_and_return; }

        } else {
            /* Even process -- send result to rank + 1 */
            err = MCA_PML_CALL(send(rbuf, count, dtype, rank + 1,
                                    MCA_COLL_BASE_TAG_ALLREDUCE,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
            if (MPI_SUCCESS != err) { goto cleanup_and_return; }
        }
    }

  cleanup_and_return:
    if (NULL != tmp_buf_raw)
        free(tmp_buf_raw);
    if (NULL != rindex)
        free(rindex);
    if (NULL != sindex)
        free(sindex);
    if (NULL != rcount)
        free(rcount);
    if (NULL != scount)
        free(scount);
    return err;
}

/*
 *   ompi_coll_base_allreduce_intra_allgather_reduce
 *
 *   Function:       use allgather for allreduce operation
 *   Accepts:        Same as MPI_Allreduce()
 *   Returns:        MPI_SUCCESS or error code
 *
 *   Description:    Implements allgather based allreduce aimed to improve internode 
 *                   allreduce latency: this method takes advantage of the send and 
 *                   receive can happen at the same time; first step is allgather
 *                   operation to allow all ranks to obtain the full dataset; the second
 *                   step is to do reduction on all ranks to get allreduce result. 
 *
 *   Limitations:    This method is designed for small message sizes allreduce because it 
 *                   is not efficient in terms of network bandwidth comparing
 *                   to gather/reduce/bcast type of approach.
 */
int ompi_coll_base_allreduce_intra_allgather_reduce(const void *sbuf, void *rbuf, size_t count,
                                                    struct ompi_datatype_t *dtype,
                                                    struct ompi_op_t *op,
                                                    struct ompi_communicator_t *comm,
                                                    mca_coll_base_module_t *module)
{
    int line = -1;
    char *partial_buf = NULL;
    char *partial_buf_start = NULL;
    char *sendtmpbuf = NULL;
    char *tmpsend = NULL;
    char *tmpsend_start = NULL;
    int err = OMPI_SUCCESS;

    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(dtype, &lb, &extent);

    int size = ompi_comm_size(comm);

    // if (ompi_comm_rank(comm) == 0) {
    //   printf("7: ALLGATHER REDUCE\n");
    //   fflush(stdout);
    // }

    sendtmpbuf = (char*) sbuf;
    if( sbuf == MPI_IN_PLACE ) {
        sendtmpbuf = (char *)rbuf;
    }
    ptrdiff_t buf_size, gap = 0;
    buf_size = opal_datatype_span(&dtype->super, (int64_t)count * size, &gap);
    partial_buf = (char *) malloc(buf_size);
    partial_buf_start = partial_buf - gap;
    buf_size = opal_datatype_span(&dtype->super, (int64_t)count, &gap);
    tmpsend = (char *) malloc(buf_size);
    tmpsend_start = tmpsend - gap;

    err = ompi_datatype_copy_content_same_ddt(dtype, count,
                                              (char*)tmpsend_start,
                                              (char*)sendtmpbuf);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    // apply allgather data so that each rank has a full copy to do reduce (trade bandwidth for better latency)
    err = comm->c_coll->coll_allgather(tmpsend_start, count, dtype,
                                       partial_buf_start, count, dtype,
                                       comm, comm->c_coll->coll_allgather_module);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    for (int target = 1; target < size; target++) {
        ompi_op_reduce(op,
                       partial_buf_start + (ptrdiff_t)target * count * extent,
                       partial_buf_start,
                       count,
                       dtype);
    }

    // move data to rbuf
    err = ompi_datatype_copy_content_same_ddt(dtype, count,
                                              (char*)rbuf,
                                              (char*)partial_buf_start);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    if (NULL != partial_buf) free(partial_buf);
    if (NULL != tmpsend) free(tmpsend);
    return MPI_SUCCESS;

err_hndl:
    if (NULL != partial_buf) {
        free(partial_buf);
        partial_buf = NULL;
        partial_buf_start = NULL;
    }
     if (NULL != tmpsend) {
        free(tmpsend);
        tmpsend = NULL;
        tmpsend_start = NULL;
    }
   OPAL_OUTPUT((ompi_coll_base_framework.framework_output,  "%s:%4d\tError occurred %d, rank %2d",
                 __FILE__, line, err, ompi_comm_rank(comm)));
    (void)line;  // silence compiler warning
    return err;

}
/* copied function (with appropriate renaming) ends here */


#define LIBSWING_MAX_STEPS 20 // With this we are ok up to 2^20 nodes, add other terms to the following arrays if needed.
static int rhos[LIBSWING_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341, 683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};

// static inline int pow_of_neg_two(int n) {
//   int power_of_two = 1 << n;
//   // If n is even, return 2^n, otherwise return -2^n
//   return (n % 2 == 0) ? power_of_two : -power_of_two;
// }

static inline int pi(int rank, int step, int comm_sz) {
  // int rho_s = (1 - pow_of_neg_two(step + 1)) / 3;
  int dest;
  if (rank % 2 == 0)  dest = (rank + rhos[step]) % comm_sz;
  else                dest = (rank - rhos[step]) % comm_sz;

  if (dest < 0) dest += comm_sz;

  return dest;
}


static inline void get_indexes_aux(int rank, int step, const int n_steps, const int adj_size, unsigned char *bitmap){
  if (step >= n_steps) return;

  int peer;
  
  for (int s = step; s < n_steps; s++){
    peer = pi(rank, s, adj_size);
    *(bitmap + peer) = 0x1;
    get_indexes_aux(peer, s + 1, n_steps, adj_size, bitmap);
  }
}


static inline void get_indexes(int rank, int step, const int n_steps, const int adj_size, unsigned char *bitmap){
  if (step >= n_steps) return;
  
  int peer = pi(rank, step, adj_size);
  *(bitmap + peer) = 0x1;
  get_indexes_aux(peer, step + 1, n_steps, adj_size, bitmap);
}


// TODO: delete this when op_thing is done
//
// static inline void my_op_is_valid(ompi_op_t * op, ompi_datatype_t * ddt)
// {
//     if (ompi_op_is_intrinsic(op)) {
//         if (ompi_datatype_is_predefined(ddt)) {
//             /* Intrinsic ddt on intrinsic op */
//             if (-1 == ompi_op_ddt_map[ddt->id] ||
//                 NULL == op->o_func.intrinsic.fns[ompi_op_ddt_map[ddt->id]]) {
//                 printf("the reduction operation %s is not defined on the %s datatype\n", op->o_name, ddt->name);
//                 return;
//             }
//         } else {
//             /* Non-intrinsic ddt on intrinsic op */
//             if ('\0' != ddt->name[0]) {
//                 printf("the reduction operation %s is not defined for non-intrinsic datatypes (attempted with datatype named \"%s\")\n",op->o_name, ddt->name);
//             } else {
//                 printf("the reduction operation %s is not defined for non-intrinsic datatypes\n", op->o_name);
//             }
//             return;
//         }
//     }
//     printf("everything ok\n");
//     return;
// }


static inline void my_reduce_copy(ompi_op_t *op, const void *source, void *target, const unsigned char *bitmap, int adj_size, const size_t small_block_count, const int split_rank, ompi_datatype_t *dtype){
  ptrdiff_t s_offset = 0, t_offset = 0, chunk_size_actual;
  size_t el_size;
  ompi_datatype_type_size(dtype, &el_size);

  for(int chunk = 0; chunk < adj_size; chunk++){
    chunk_size_actual = (chunk < split_rank) ? (ptrdiff_t) ((small_block_count + 1) * el_size) : (ptrdiff_t) (small_block_count * el_size);
    if (bitmap[chunk] != 0){
      ompi_op_reduce(op, (char *)source + s_offset, (char *)target + t_offset, (chunk < split_rank) ? small_block_count + 1 : small_block_count, dtype);
      s_offset += chunk_size_actual;
    }
    t_offset += chunk_size_actual;
  }
}


static inline void my_reduce_indexed_dtype(ompi_op_t *op, const void *source, void *target, const unsigned char *bitmap, int adj_size, const size_t small_block_count, const int split_rank, ompi_datatype_t *dtype){
  ptrdiff_t offset = 0, chunk_size_actual;
  size_t el_size;
  ompi_datatype_type_size(dtype, &el_size);

  for(int chunk = 0; chunk < adj_size; chunk++){
    chunk_size_actual = (chunk < split_rank) ? (ptrdiff_t) ((small_block_count + 1) * el_size) : (ptrdiff_t) (small_block_count * el_size);
    if (bitmap[chunk] != 0){
      ompi_op_reduce(op, (char *)source + offset, (char *)target + offset, (chunk < split_rank) ? small_block_count + 1 : small_block_count, dtype);
    }
    offset += chunk_size_actual;
  }
}


// static inline void my_reduce(ompi_op_t *op, const void *source, void *target, const unsigned char *bitmap, int adj_size, const size_t small_block_count, const int split_rank, ompi_datatype_t *dtype){
//   // NOTE: try to use ompi_datatype_get_single_predefined_type_from_args(dtype)
//
//   if(ompi_datatype_is_predefined(dtype)){
//     my_reduce_copy(op, source, target, bitmap, adj_size, small_block_count, split_rank, dtype);
//     return;
//   }
//   else{
//     
//     my_reduce_indexed_dtype(op, source, target, bitmap, adj_size, small_block_count, split_rank, ompi_datatype_get_single_predefined_type_from_args(dtype));
//     return;
//   }
// }


static inline void copy_chunks(const void *source, void *target, const unsigned char *bitmap, int adj_size, const size_t small_block_count, const int split_rank, ompi_datatype_t *dtype) {
  ptrdiff_t s_offset = 0, t_offset = 0;
  size_t el_size, chunk_size_actual;
  ompi_datatype_type_size(dtype, &el_size);

  for (int chunk = 0; chunk < adj_size; chunk++) {
    chunk_size_actual = (chunk < split_rank) ? (small_block_count + 1) * el_size : small_block_count * el_size;
    if (bitmap[chunk] != 0) {
      memcpy((char *)target + t_offset, (char *)source + s_offset, chunk_size_actual);
      t_offset += (ptrdiff_t) chunk_size_actual;
    }
    s_offset += (ptrdiff_t) chunk_size_actual;
  }
}

static inline void my_overwrite(const void *source, void *target, const unsigned char *bitmap, int adj_size, const size_t small_block_count, const int split_rank, struct ompi_datatype_t *dtype){
  ptrdiff_t s_offset = 0, t_offset = 0;
  size_t el_size, chunk_size_actual;
  ompi_datatype_type_size(dtype, &el_size);

  for(int chunk = 0; chunk < adj_size; chunk++){
    chunk_size_actual = (chunk < split_rank) ? (small_block_count + 1) * el_size : small_block_count * el_size;
    if (bitmap[chunk] != 0){
      memcpy((char *)target + t_offset, (char *)source + s_offset, chunk_size_actual);
      s_offset += (ptrdiff_t) chunk_size_actual;
    }
    t_offset += (ptrdiff_t) chunk_size_actual;
  }
}


static inline int indexed_datatype(ompi_datatype_t **new_dtype, const unsigned char *bitmap, int adj_size, int w_size, const size_t small_block_count, const int split_rank, ompi_datatype_t *old_dtype, int *block_len, int *disp){
  int index = 0, disp_counter = 0;
  for (int i = 0; i < adj_size; i++){
    if (bitmap[i] != 0){
      block_len[index] =  i < split_rank ? (int) (small_block_count + 1) : (int) small_block_count;
      disp[index] = disp_counter;
      index++;
    }
    disp_counter += i < split_rank ? (int) (small_block_count + 1): (int) small_block_count;
  }

  if (index != w_size){
    fprintf(stderr, "\n\nERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\nindex!=w_size ---> i:%d w_s:%d\n\n", index, w_size);
    return MPI_ERR_UNKNOWN;
  }

  ompi_datatype_create_indexed(w_size, block_len, disp, old_dtype, new_dtype);
  { 
    const int* a_i[3] = {&w_size, block_len, disp};
    ompi_datatype_set_args( *new_dtype, 2 * w_size + 1, a_i, 0, NULL, 1, &old_dtype, MPI_COMBINER_INDEXED );
  }
  ompi_datatype_commit(new_dtype);

  return MPI_SUCCESS;
}


static inline int get_static_bitmap(const int** send_bitmap, const int** recv_bitmap, int n_steps, int comm_sz, int rank) {
  // verify that comm_sz is exactly 2^n_steps
  if (comm_sz != (1 << n_steps)){
    return -1;
  }
  // Static bitmaps are defined up to 256 ranks, so since n_steps = log2 comm_sz -> log2(256)=8
  if (n_steps < 1 || n_steps > 8) {
    return -1;
  }

  *send_bitmap = ((const int*)static_send_bitmaps[n_steps]) + (ptrdiff_t)(rank * n_steps);
  *recv_bitmap = ((const int*)static_recv_bitmaps[n_steps]) + (ptrdiff_t)(rank * n_steps);

  return 0;  // Success
}

int ompi_coll_base_allreduce_swing(const void *send_buf, void *recv_buf, size_t count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module) {
  int rank, size;
  int ret, line; // for error handling
  char *tmpsend, *tmprecv, *tmpswap, *inplacebuf_free = NULL;
  ptrdiff_t span, gap = 0;
  

  size = ompi_comm_size(comm);
  rank = ompi_comm_rank(comm);

  // if (rank == 0){
  //   printf("8: SWING LATENCY OPTIMAL\n");
  //   fflush(stdout);
  // }

  OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "coll:base:allreduce_swing rank %d", rank));

  // Special case for size == 1
  if (1 == size) {
    if (MPI_IN_PLACE != send_buf) {
      ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)recv_buf, (char*)send_buf);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
    }
    return MPI_SUCCESS;
  }
  
  // Allocate and initialize temporary send buffer
  span = opal_datatype_span(&dtype->super, count, &gap);
  inplacebuf_free = (char*) malloc(span + gap);
  char *inplacebuf = inplacebuf_free + gap;

  // Copy content from send_buffer to inplacebuf
  if (MPI_IN_PLACE == send_buf) {
      ret = ompi_datatype_copy_content_same_ddt(dtype, count, inplacebuf, (char*)recv_buf);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  } else {
      ret = ompi_datatype_copy_content_same_ddt(dtype, count, inplacebuf, (char*)send_buf);
      if (ret < 0) { line = __LINE__; goto error_hndl; }
  }


  tmpsend = inplacebuf;
  tmprecv = (char*) recv_buf;
  
  int adjsize, extra_ranks, new_rank = rank; // needed to handle not power of 2 cases

  //Determine nearest power of two less than or equal to size
  int steps = opal_hibit(size, comm->c_cube_dim + 1);
  if (-1 == steps) {
    return MPI_ERR_ARG;
  }
  adjsize = 1 << steps;

  //Number of nodes that exceed max(2^n)< size
  extra_ranks = size - adjsize;
  int is_power_of_two = size  == adjsize;

  // First part of computation to get a 2^n number of nodes.
  // What happens is that first #extra_rank even nodes sends their
  // data to the successive node and do not partecipate in the general
  // collective call operation.
  // All the nodes that do not stop their computation will receive an alias
  // called new_node, used to calculate their correct destination wrt this
  // new "cut" topology.
  if (rank <  (2 * extra_ranks)) {
    if (0 == (rank % 2)) {
      ret = MCA_PML_CALL(send(tmpsend, count, dtype, (rank + 1), MCA_COLL_BASE_TAG_ALLREDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      new_rank = -1;
    } else {
      ret = MCA_PML_CALL(recv(tmprecv, count, dtype, (rank - 1), MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE));
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      ompi_op_reduce(op, tmprecv, tmpsend, count, dtype);
      new_rank = rank >> 1;
    }
  } else new_rank = rank - extra_ranks;
  
  
  // Actual allreduce computation for general cases
  int s, vdest, dest;
  for (s = 0; s < steps; s++){
    if (new_rank < 0) break;
    vdest = pi(new_rank, s, adjsize);

    dest = is_power_of_two ? vdest : (vdest < extra_ranks) ? (vdest << 1) + 1 : vdest + extra_ranks;

    ret = ompi_coll_base_sendrecv_actual(tmpsend, count, dtype, dest, MCA_COLL_BASE_TAG_ALLREDUCE, tmprecv, count, dtype, dest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE);
    if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    
    if (rank < dest) {
      ompi_op_reduce(op, tmpsend, tmprecv, count, dtype);
      tmpswap = tmprecv;
      tmprecv = tmpsend;
      tmpsend = tmpswap;
    } else {
      ompi_op_reduce(op, tmprecv, tmpsend, count, dtype);
    }
  }
  
  // Final results is sent to nodes that are not included in general computation
  // (general computation loop requires 2^n nodes).
  if (rank < (2 * extra_ranks)){
    if (new_rank < 0){
      ret = MCA_PML_CALL(recv(recv_buf, count, dtype, (rank + 1), MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE));
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
      tmpsend = (char*)recv_buf;
    } else {
      ret = MCA_PML_CALL(send(tmpsend, count, dtype, (rank - 1), MCA_COLL_BASE_TAG_ALLREDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
      if (MPI_SUCCESS != ret) { line = __LINE__; goto error_hndl; }
    }
  }

  if (tmpsend != recv_buf) {
    ret = ompi_datatype_copy_content_same_ddt(dtype, count, (char*) recv_buf, tmpsend);
    if (ret < 0) { line = __LINE__; goto error_hndl; }
  }

  free(inplacebuf_free);
  return MPI_SUCCESS;

  error_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "%s:%4d\tRank %d Error occurred %d\n",
                 __FILE__, line, rank, ret));
    (void)line;  // silence compiler warning
    if (NULL != inplacebuf_free) free(inplacebuf_free);
    return ret;
}


int ompi_coll_base_allreduce_swing_rabenseifner_memcpy(
    const void *send_buf, void *recv_buf, size_t count, struct ompi_datatype_t *dtype,
    struct ompi_op_t *op, struct ompi_communicator_t *comm,
    mca_coll_base_module_t *module)
{
  int comm_sz, rank; 
  comm_sz = ompi_comm_size(comm);
  rank = ompi_comm_rank(comm);
  
  // if (rank == 0) {
  //   printf("9: SWING RABENSEIFNER MEMCPY\n");
  //   fflush(stdout);
  // }

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to comm_sz,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if comm_sz != adj_size
  int n_steps, adj_size;
  n_steps = opal_hibit(comm_sz, comm->c_cube_dim + 1);
  adj_size = 1 << n_steps;
  
  //WARNING: Assuming comm_sz is a pow of 2
  int vrank, vdest;
  vrank = rank;
  
  ptrdiff_t lb, extent, gap = 0;
  ompi_datatype_get_extent(dtype, &lb, &extent);
  
  int split_rank;
  size_t small_block_count, large_block_count;
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, adj_size, split_rank, large_block_count, small_block_count);
  
  // Find the biggest power-of-two smaller than count to allocate as few memory as necessary for buffers
  int max_bit_pos, n_pow;
  max_bit_pos = (int) (sizeof(count) * CHAR_BIT) - 1;
  n_pow = opal_hibit((int)count, max_bit_pos);  // WARNING: here count is casted to int, what happens if count>MAX_INT? 
  size_t buf_count = 1 << n_pow;
  ptrdiff_t buf_size = opal_datatype_span(&dtype->super, buf_count, &gap);

  // Temporary target buffer for send operations and source buffer for reduce and overwrite operations
  char *tmp_buf_raw, *tmp_buf, *cp_buf_raw, *cp_buf;
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;
  // Temporary target buffer for copy operations and source buffer for send operations
  cp_buf_raw = (char *)malloc(buf_size);
  cp_buf = cp_buf_raw - gap;
  

  // Copy into receive_buffer content of send_buffer to not produce side effects on send_buffer
  if (send_buf != MPI_IN_PLACE) {
    ompi_datatype_copy_content_same_ddt(dtype, count, (char *)recv_buf, (char *)send_buf);
  }
  
  unsigned char *s_bitmap = NULL, *r_bitmap = NULL;
  int bitmap_offset = 0;
  s_bitmap = (unsigned char *) calloc(adj_size * n_steps, sizeof(unsigned char));
  r_bitmap = (unsigned char *) calloc(adj_size * n_steps, sizeof(unsigned char));
  
  size_t send_count, recv_count;
  int step;
  // Reduce-Scatter phase
  for (step = 0; step < n_steps; step++) {
    vdest = pi(vrank, step, adj_size);
    
    get_indexes(vrank, step, n_steps, adj_size, s_bitmap + bitmap_offset);
    get_indexes(vdest, step, n_steps, adj_size, r_bitmap + bitmap_offset);
    
    copy_chunks(recv_buf, cp_buf, s_bitmap + bitmap_offset, adj_size, small_block_count, split_rank, dtype);   

    send_count = 0;
    recv_count = 0;
    for(int i = 0; i < adj_size; i++){
      if(s_bitmap[i + bitmap_offset] != 0)       send_count += (i < split_rank) ? large_block_count : small_block_count;
      else if(r_bitmap[i + bitmap_offset] != 0)  recv_count += (i < split_rank) ? large_block_count : small_block_count;
    }

    ompi_coll_base_sendrecv(cp_buf, send_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, tmp_buf, recv_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);
    
    my_reduce_copy(op, tmp_buf, recv_buf, r_bitmap + bitmap_offset, adj_size, small_block_count, split_rank, dtype);
 
    bitmap_offset += adj_size;
  }
  
  // Allgather phase
  bitmap_offset -= adj_size;
  for(step = n_steps - 1; step >= 0; step--) {
    vdest = pi(vrank, step, adj_size);

    copy_chunks(recv_buf, cp_buf, r_bitmap + bitmap_offset, adj_size, small_block_count, split_rank, dtype);   
    
    send_count = 0;
    recv_count = 0;
    for(int i = 0; i < adj_size; i++){
      if(s_bitmap[i + bitmap_offset] != 0)       send_count += (i < split_rank) ? large_block_count : small_block_count;
      else if(r_bitmap[i + bitmap_offset] != 0)  recv_count += (i < split_rank) ? large_block_count : small_block_count;
    }
    
    ompi_coll_base_sendrecv(cp_buf, recv_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, tmp_buf, send_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);
    
    my_overwrite(tmp_buf, recv_buf, s_bitmap + bitmap_offset, adj_size, small_block_count, split_rank, dtype);

    bitmap_offset -= adj_size;
  }
 
  free(s_bitmap);
  free(r_bitmap);

  free(tmp_buf_raw);
  free(cp_buf_raw);

  return MPI_SUCCESS;
}


int ompi_coll_base_allreduce_swing_rabenseifner_dt(
    const void *send_buf, void *recv_buf, size_t count, struct ompi_datatype_t *dtype,
    struct ompi_op_t *op, struct ompi_communicator_t *comm,
    mca_coll_base_module_t *module)
{
  int comm_sz, rank;
  comm_sz = ompi_comm_size(comm);
  rank = ompi_comm_rank(comm);
  
  // if (rank == 0) {
  //   printf("10: SWING RABENSEIFNER DT\n");
  //   fflush(stdout);
  // }

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to comm_sz,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if comm_sz != adj_size
  int n_steps, adj_size;
  n_steps = opal_hibit(comm_sz, comm->c_cube_dim + 1);
  adj_size = 1 << n_steps;


  //WARNING: Assuming comm_sz is a pow of 2
  int vrank, vdest;
  vrank = rank;
  
  int split_rank;
  size_t small_block_count, large_block_count;
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, adj_size, split_rank, large_block_count, small_block_count);

  // NOTE: what do I do with this? I think extent can be used instead of size, need to control
  // also strided datatipes can use extent and true extent
  ptrdiff_t lb, extent, gap = 0;
  ompi_datatype_get_extent(dtype, &lb, &extent);
  
  // Temporary target buffer for send operations and source buffer for reduce and overwrite operations
  char *tmp_buf_raw, *tmp_buf;
  ptrdiff_t buf_size = opal_datatype_span(&dtype->super, count, &gap);
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;
  
  // Copy into receive_buffer content of send_buffer to not produce side effects on send_buffer
  if (send_buf != MPI_IN_PLACE) {
    ompi_datatype_copy_content_same_ddt(dtype, count, (char *)recv_buf, (char *)send_buf);
  }

  unsigned char *s_bitmap = NULL, *r_bitmap = NULL;
  int bitmap_offset = 0;
  s_bitmap = (unsigned char *) calloc(adj_size * n_steps, sizeof(unsigned char));
  r_bitmap = (unsigned char *) calloc(adj_size * n_steps, sizeof(unsigned char));
  
  ompi_datatype_t **ind_dtype;
  int *block_len, *disp, dtype_offset = 0;
  ind_dtype = (ompi_datatype_t **) malloc(2 * n_steps * sizeof(*ind_dtype));
  block_len = (int *)malloc(adj_size * sizeof(int));
  disp = (int *)malloc(adj_size * sizeof(int));
  
  int step;
  size_t w_size, el_size;
  w_size = (size_t) adj_size;
  ompi_datatype_type_size(dtype, &el_size);

  // Reduce-Scatter phase
  for (step = 0; step < n_steps; step++) {
    w_size >>= 1;

    vdest = pi(vrank, step, adj_size);
    
    get_indexes(vrank, step, n_steps, adj_size, s_bitmap + bitmap_offset);
    get_indexes(vdest, step, n_steps, adj_size, r_bitmap + bitmap_offset);
    
    ind_dtype[0 + dtype_offset] = MPI_DATATYPE_NULL;
    indexed_datatype(&ind_dtype[0 + dtype_offset], s_bitmap + bitmap_offset, adj_size, w_size, small_block_count, split_rank, dtype, block_len, disp);
    ind_dtype[1 + dtype_offset] = MPI_DATATYPE_NULL;
    indexed_datatype(&ind_dtype[1 + dtype_offset], r_bitmap + bitmap_offset, adj_size, w_size, small_block_count, split_rank, dtype, block_len, disp);
     
    ompi_coll_base_sendrecv(recv_buf, 1, ind_dtype[0 + dtype_offset], vdest, MCA_COLL_BASE_TAG_ALLREDUCE, tmp_buf, 1, ind_dtype[1 + dtype_offset], vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);

    my_reduce_indexed_dtype(op, tmp_buf, recv_buf, r_bitmap + bitmap_offset, adj_size, small_block_count, split_rank, dtype);

    bitmap_offset += adj_size;
    dtype_offset += 2;
    
    // TODO: custom_op
    //
    // size_t recv_count = 0;
    // for(int i = 0; i < adj_size; i++){
    //   if(r_bitmap[i + bitmap_offset] != 0)  recv_count += (i < split_rank) ? large_block_count : small_block_count;
    // }
    // ompi_op_t* custom_op = ompi_op_create_user(ompi_op_is_commute(op), op->o_func.fort_fn);
    // my_op_is_valid(custom_op, ind_dtype[1+dtype_offset]);
    // ompi_op_reduce(custom_op, tmp_buf, recv_buf, 1, ind_dtype[1 + dtype_offset]);
  }
  
  // Allgather phase
  bitmap_offset -= adj_size;
  dtype_offset -= 2;
  for(step = n_steps - 1; step >= 0; step--) {
    vdest = pi(vrank, step, adj_size);

    ompi_coll_base_sendrecv(recv_buf, 1, ind_dtype[1 + dtype_offset], vdest, MCA_COLL_BASE_TAG_ALLREDUCE, recv_buf, 1, ind_dtype[0 + dtype_offset], vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);
    
    ompi_datatype_destroy(&ind_dtype[1 + dtype_offset]);
    ompi_datatype_destroy(&ind_dtype[0 + dtype_offset]);
    
    w_size <<= 1;
    bitmap_offset -= adj_size;
    dtype_offset -= 2;
  }

  free(ind_dtype);
  free(block_len);
  free(disp);

  free(s_bitmap);
  free(r_bitmap);

  free(tmp_buf_raw);


  return MPI_SUCCESS;
}


int ompi_coll_base_allreduce_swing_rabenseifner_dt_single(
    const void *send_buf, void *recv_buf, size_t count, struct ompi_datatype_t *dtype,
    struct ompi_op_t *op, struct ompi_communicator_t *comm,
    mca_coll_base_module_t *module)
{ 

  int comm_sz, rank;
  comm_sz = ompi_comm_size(comm);
  rank = ompi_comm_rank(comm);
  
  // if (rank == 0) {
  //   printf("11: SWING RABENSEIFNER DT SINGLE\n");
  //   fflush(stdout);
  // }

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to comm_sz,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if comm_sz != adj_size
  int n_steps, adj_size;
  n_steps = opal_hibit(comm_sz, comm->c_cube_dim + 1);
  adj_size = 1 << n_steps;


  //WARNING: Assuming comm_sz is a pow of 2
  int vrank, vdest;
  vrank = rank;
  
  int split_rank;
  size_t small_block_count, large_block_count;
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, adj_size, split_rank, large_block_count, small_block_count);
  
  // NOTE: what do I do with this? I think extent can be used instead of size, need to control
  // also strided datatipes can use extent and true extent
  ptrdiff_t lb, extent, gap = 0;
  ompi_datatype_get_extent(dtype, &lb, &extent);

  // Find the biggest power-of-two smaller than count to allocate as few memory as necessary for buffers
  int max_bit_pos, n_pow;
  max_bit_pos = (int) (sizeof(count) * CHAR_BIT) - 1;
  n_pow = opal_hibit((int)count, max_bit_pos);  // WARNING: here count is casted to int, what happens if count>MAX_INT? 
  size_t buf_count = 1 << n_pow;
  ptrdiff_t buf_size = opal_datatype_span(&dtype->super, buf_count, &gap);

  // Temporary target buffer for send operations and source buffer for reduce and overwrite operations
  char *tmp_buf_raw, *tmp_buf;
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;
  
  // Copy into receive_buffer content of send_buffer to not produce side effects on send_buffer
  if (send_buf != MPI_IN_PLACE) {
    ompi_datatype_copy_content_same_ddt(dtype, count, (char *)recv_buf, (char *)send_buf);
  }

  unsigned char *s_bitmap = NULL, *r_bitmap = NULL;
  int recv_count, bitmap_offset = 0;
  s_bitmap = calloc(adj_size * n_steps, sizeof(unsigned char));
  r_bitmap = calloc(adj_size * n_steps, sizeof(unsigned char));


  int *block_len, *disp;
  ompi_datatype_t *ind_dtype = MPI_DATATYPE_NULL;
  block_len = (int *)malloc(adj_size * sizeof(int));
  disp = (int *)malloc(adj_size * sizeof(int));
  
  int step;
  size_t w_size = (size_t) adj_size;
  // Reduce-Scatter phase
  for (step = 0; step < n_steps; step++) {
    w_size >>= 1;

    vdest = pi(vrank, step, adj_size);
    
    get_indexes(vrank, step, n_steps, adj_size, s_bitmap + bitmap_offset);
    get_indexes(vdest, step, n_steps, adj_size, r_bitmap + bitmap_offset);
    
    indexed_datatype(&ind_dtype, s_bitmap + bitmap_offset, adj_size, w_size, small_block_count, split_rank, dtype, block_len, disp);
    
    recv_count = 0;
    for(int i = 0; i < adj_size; i++){
      if(r_bitmap[i + bitmap_offset] != 0)  recv_count += (i < split_rank) ? large_block_count : small_block_count;
    }

    ompi_coll_base_sendrecv(recv_buf, 1, ind_dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, tmp_buf, recv_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);
    
    my_reduce_copy(op, tmp_buf, recv_buf, r_bitmap + bitmap_offset, adj_size, small_block_count, split_rank, dtype);

    bitmap_offset += adj_size;
    
    ompi_datatype_destroy(&ind_dtype);
    ind_dtype = MPI_DATATYPE_NULL;
  }
  
  // Allgather phase
  bitmap_offset -= adj_size;
  for(step = n_steps - 1; step >= 0; step--) {
    vdest = pi(vrank, step, adj_size);

    indexed_datatype(&ind_dtype, r_bitmap + bitmap_offset, adj_size, w_size, small_block_count, split_rank, dtype, block_len, disp);
    
    recv_count = 0;
    for(int i = 0; i < adj_size; i++){
      if(s_bitmap[i + bitmap_offset] != 0)  recv_count += (i < split_rank) ? large_block_count : small_block_count;
    }

    ompi_coll_base_sendrecv(recv_buf, 1, ind_dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, tmp_buf, recv_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);
    
    my_overwrite(tmp_buf, recv_buf, s_bitmap + bitmap_offset, adj_size, small_block_count, split_rank, dtype);
    
    w_size <<= 1;
    bitmap_offset -= adj_size;
    ompi_datatype_destroy(&ind_dtype);
    ind_dtype = MPI_DATATYPE_NULL;
  }

  free(s_bitmap);
  free(r_bitmap);

  free(tmp_buf_raw);

  return MPI_SUCCESS;
}



int ompi_coll_base_allreduce_swing_rabenseifner_segmented(const void *send_buf, void *recv_buf, size_t count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module, uint32_t segsize)
{ 
  int comm_sz, rank;
  comm_sz = ompi_comm_size(comm);
  rank = ompi_comm_rank(comm);
  
  // if (rank == 0) {
  //   printf("12: SWING RABENSEIFNER SEGMENTED\n");
  //   fflush(stdout);
  // }

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to comm_sz,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if comm_sz != adj_size
  int n_steps, adj_size;
  n_steps = opal_hibit(comm_sz, comm->c_cube_dim + 1);
  adj_size = 1 << n_steps;
  
  //WARNING: Assuming comm_sz is a pow of 2
  int vrank, vdest;
  vrank = rank;

  int split_rank;
  size_t small_block_count, large_block_count;
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, adj_size, split_rank, large_block_count, small_block_count);
  
  size_t typelng, seg_count = large_block_count;
  ompi_datatype_type_size(dtype, &typelng);
  COLL_BASE_COMPUTED_SEGCOUNT(segsize, typelng, seg_count);

  int num_phases = (int) (large_block_count / seg_count);
  if ((large_block_count % seg_count) != 0) num_phases++;
  
  ompi_request_t *reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  
  int k, inbi;
  size_t max_seg_count;
  ptrdiff_t lb, extent, gap;
  COLL_BASE_COMPUTE_BLOCKCOUNT(large_block_count, num_phases, inbi, max_seg_count, k);
  ompi_datatype_get_extent(dtype, &lb, &extent);
  ptrdiff_t max_real_segsize = opal_datatype_span(&dtype->super, max_seg_count, &gap);

  /* Allocate and initialize temporary buffers */
  char *tmp_send = NULL, *tmp_recv = NULL;
  char *tmp_buf[2] = {NULL, NULL};
  tmp_buf[0] = (char *) malloc(max_real_segsize);
  tmp_buf[1] = (char *) malloc(max_real_segsize);

  // Copy into receive_buffer content of send_buffer to not produce side effects on send_buffer
  if (send_buf != MPI_IN_PLACE) {
    ompi_datatype_copy_content_same_ddt(dtype, count, (char *)recv_buf, (char *)send_buf);
  }
  
  unsigned char *s_bitmap = NULL, *r_bitmap = NULL;
  int bitmap_offset = 0;
  s_bitmap = (unsigned char *) calloc(adj_size * n_steps, sizeof(unsigned char));
  r_bitmap = (unsigned char *) calloc(adj_size * n_steps, sizeof(unsigned char));

  int step, s_ind, r_ind, s_split_seg, r_split_seg, seg_ind;
  size_t s_block_count, r_block_count;
  size_t s_large_seg_count, s_small_seg_count, r_large_seg_count, r_small_seg_count;
  size_t r_count, s_count, prev_r_count;
  ptrdiff_t s_block_offset, r_block_offset, r_seg_offset, s_seg_offset;
  // Reduce-Scatter phase
  for (step = 0; step < n_steps; step++) {
    vdest = pi(vrank, step, adj_size);
    
    get_indexes(vrank, step, n_steps, adj_size, s_bitmap + bitmap_offset);
    get_indexes(vdest, step, n_steps, adj_size, r_bitmap + bitmap_offset);

    s_ind = 0;
    r_ind = 0;
    while (s_ind < adj_size && r_ind < adj_size) {
      // Navigate send and recv bitmap to find first block to send and recv
      while (s_ind < adj_size && s_bitmap[s_ind + bitmap_offset] != 1) { s_ind++;}
      while (r_ind < adj_size && r_bitmap[r_ind + bitmap_offset] != 1) { r_ind++;}
      
      // Scatter reduce the block
      if (r_ind < adj_size && s_ind < adj_size) {
        inbi = 0;
        
        // For each one of send block and recv block calculate:
        // - block_count: number of elements in the block
        // - large_seg_count, small_seg_count: number of elements in big and small segments
        // - split_seg: indicates the first of the small segments
        s_block_count = (s_ind < split_rank) ? large_block_count : small_block_count;
        r_block_count = (r_ind < split_rank) ? large_block_count : small_block_count;
        COLL_BASE_COMPUTE_BLOCKCOUNT(s_block_count, num_phases, s_split_seg, s_large_seg_count, s_small_seg_count);
        COLL_BASE_COMPUTE_BLOCKCOUNT(r_block_count, num_phases, r_split_seg, r_large_seg_count, r_small_seg_count);

        // Calculate the offset of the send and recv block wrt buffer (in bytes)
        s_block_offset = (s_ind < split_rank) ? ((ptrdiff_t) s_ind * (ptrdiff_t) large_block_count) * extent :
                          ((ptrdiff_t) s_ind * (ptrdiff_t) small_block_count + split_rank) * extent;
        r_block_offset = (r_ind < split_rank) ? ((ptrdiff_t) r_ind * (ptrdiff_t) large_block_count) * extent : 
                          ((ptrdiff_t) r_ind * (ptrdiff_t) small_block_count + split_rank) * extent;

        // Post an irecv for the first segment
        MCA_PML_CALL(irecv(tmp_buf[inbi], r_large_seg_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, &reqs[inbi]));
        
        // Send the first segment
        tmp_send = (char *)recv_buf + s_block_offset;
        MCA_PML_CALL(send(tmp_send, s_large_seg_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
        
        for(seg_ind = 1; seg_ind < num_phases; seg_ind++){
          inbi = inbi ^ 0x1;
          
          // Post an irecv for the current segment (i.e. seg[seg_ind])
          r_count = (seg_ind < r_split_seg) ? r_large_seg_count: r_small_seg_count;
          MCA_PML_CALL(irecv(tmp_buf[inbi], r_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, &reqs[inbi]));

          // Wait for the arrival of the previous block
          ompi_request_wait(&reqs[inbi ^ 0x1], MPI_STATUS_IGNORE);
          
          // Calculate the offset of the recv segment wrt the start of the block (in bytes) and the number of elements of the previous recv
          r_seg_offset = (seg_ind - 1 < r_split_seg) ? ((ptrdiff_t) (seg_ind - 1) * (ptrdiff_t) r_large_seg_count) * extent :
                                                       (((ptrdiff_t) (seg_ind - 1) * (ptrdiff_t) r_small_seg_count) + (ptrdiff_t) r_split_seg) * extent;
          tmp_recv = (char *) recv_buf + (r_block_offset + r_seg_offset);
          prev_r_count = ((seg_ind - 1) < r_split_seg) ? r_large_seg_count : r_small_seg_count;

          // Reduce the previous block
          ompi_op_reduce(op, tmp_buf[inbi ^ 0x1], tmp_recv, prev_r_count, dtype);

          // Calculate offset and count of the current block and send it
          s_seg_offset = (seg_ind < s_split_seg) ? ((ptrdiff_t) seg_ind * (ptrdiff_t) s_large_seg_count) * extent :
                                                   (((ptrdiff_t) seg_ind * (ptrdiff_t) s_small_seg_count) + (ptrdiff_t) s_split_seg) * extent;
          tmp_send += s_seg_offset;
          s_count = (seg_ind < s_split_seg) ? s_large_seg_count: s_small_seg_count;
          MCA_PML_CALL(send(tmp_send, s_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
        }
        // Wait for the last segment to arrive
        ompi_request_wait(&reqs[inbi], MPI_STATUS_IGNORE);

        // Reduce the last segment
        r_seg_offset = (((ptrdiff_t) (num_phases - 1) * (ptrdiff_t) r_small_seg_count) + (ptrdiff_t) r_split_seg) * extent;
        tmp_recv = (char*) recv_buf + (r_block_offset + r_seg_offset);
        ompi_op_reduce(op, tmp_buf[inbi], tmp_recv, r_small_seg_count, dtype);
      }
      s_ind++;
      r_ind++;
    }
    bitmap_offset += adj_size;
  }
  
  
  // Allgather phase
  ompi_datatype_t *s_ind_dtype = MPI_DATATYPE_NULL, *r_ind_dtype = MPI_DATATYPE_NULL;
  int *block_len, *disp;
  block_len = (int *)malloc(adj_size * sizeof(int));
  disp = (int *)malloc(adj_size * sizeof(int));
  bitmap_offset -= adj_size;
  size_t w_size = 1;
  for(step = n_steps - 1; step >= 0; step--) {
    vdest = pi(vrank, step, adj_size);
    
    indexed_datatype(&s_ind_dtype, s_bitmap + bitmap_offset, adj_size, w_size, small_block_count, split_rank, dtype, block_len, disp);
    indexed_datatype(&r_ind_dtype, r_bitmap + bitmap_offset, adj_size, w_size, small_block_count, split_rank, dtype, block_len, disp);

    ompi_coll_base_sendrecv(recv_buf, 1, r_ind_dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, recv_buf, 1, s_ind_dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);
    
    ompi_datatype_destroy(&s_ind_dtype);
    s_ind_dtype = MPI_DATATYPE_NULL;
    ompi_datatype_destroy(&r_ind_dtype);
    r_ind_dtype = MPI_DATATYPE_NULL;
    
    w_size <<= 1;
    bitmap_offset -= adj_size;
  }
 
  free(s_bitmap);
  free(r_bitmap);

  free(block_len);
  free(disp);

  free(tmp_buf[0]);
  free(tmp_buf[1]);

  return MPI_SUCCESS;
}


int ompi_coll_base_allreduce_swing_rabenseifner_contiguous(const void *send_buf, void *recv_buf, size_t count, struct ompi_datatype_t *dtype, struct ompi_op_t *op, struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
  int comm_sz, rank; 
  comm_sz = ompi_comm_size(comm);
  rank = ompi_comm_rank(comm);
  
  // if (rank == 0) {
  //   printf("13: SWING RABENSEIFNER CONTIGUOUS\n");
  //   fflush(stdout);
  // }

  // Find number of steps of scatter-reduce and allgather,
  // biggest power of two smaller or equal to comm_sz,
  // size of send_window (number of chunks to send/recv at each step)
  // and alias of the rank to be used if comm_sz != adj_size
  int n_steps, adj_size;
  n_steps = opal_hibit(comm_sz, comm->c_cube_dim + 1);
  adj_size = 1 << n_steps;
  
  //WARNING: Assuming comm_sz is a pow of 2
  int vrank, vdest;
  vrank = rank;
  
  ptrdiff_t lb, extent, gap = 0;
  ompi_datatype_get_extent(dtype, &lb, &extent);
  
  int split_rank;
  size_t small_block_count, large_block_count;
  COLL_BASE_COMPUTE_BLOCKCOUNT(count, adj_size, split_rank, large_block_count, small_block_count);
  
  // Find the biggest power-of-two smaller than count to allocate as few memory as necessary for buffers
  int max_bit_pos, n_pow;
  max_bit_pos = (int) (sizeof(count) * CHAR_BIT) - 1;
  n_pow = opal_hibit((int)count, max_bit_pos);  // WARNING: here count is casted to int, what happens if count>MAX_INT? 
  size_t buf_count = 1 << n_pow;
  ptrdiff_t buf_size = opal_datatype_span(&dtype->super, buf_count, &gap);

  // Temporary target buffer for send operations and source buffer for reduce and overwrite operations
  char *tmp_send = NULL, *tmp_recv = NULL;
  char *tmp_buf_raw, *tmp_buf;
  tmp_buf_raw = (char *)malloc(buf_size);
  tmp_buf = tmp_buf_raw - gap;

  // Copy into receive_buffer content of send_buffer to not produce side effects on send_buffer
  if (send_buf != MPI_IN_PLACE) {
    ompi_datatype_copy_content_same_ddt(dtype, count, (char *)recv_buf, (char *)send_buf);
  }
  
  const int *s_bitmap = NULL, *r_bitmap = NULL;

  if(get_static_bitmap(&s_bitmap, &r_bitmap, n_steps, comm_sz, rank) == -1){
    free(tmp_buf);
    return MPI_ERR_UNKNOWN;
  }
  
  int step, w_size = adj_size;
  size_t s_count, r_count;
  ptrdiff_t s_offset, r_offset;
  // Reduce-Scatter phase
  for (step = 0; step < n_steps; step++) {
    w_size >>= 1;
    vdest = pi(vrank, step, adj_size);

    s_count = (s_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * large_block_count :
                  (s_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_block_count :
                    (size_t)w_size * small_block_count + (size_t)(split_rank - s_bitmap[step]);
    s_offset = (s_bitmap[step] <= split_rank) ?
                (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(large_block_count * extent) :
                (ptrdiff_t)(s_bitmap[step] * (int) small_block_count + split_rank) * (ptrdiff_t) extent;

    r_count = (r_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * large_block_count :
                  (r_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_block_count :
                    (size_t)w_size * small_block_count + (size_t)(split_rank - r_bitmap[step]);
    r_offset = (r_bitmap[step] <= split_rank) ?
                (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(large_block_count * extent) :
                (ptrdiff_t)(r_bitmap[step] * (int)small_block_count + split_rank) * (ptrdiff_t) extent;
    
    tmp_send = (char *)recv_buf + s_offset;
    ompi_coll_base_sendrecv(tmp_send, s_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, tmp_buf, r_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);
    
    tmp_recv = (char *) recv_buf + r_offset;
    ompi_op_reduce(op, tmp_buf, tmp_recv, r_count, dtype);
  }
  
  // Allgather phase
  for(step = n_steps - 1; step >= 0; step--) {
    vdest = pi(vrank, step, adj_size);
    
    s_count = (s_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * large_block_count :
                  (s_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_block_count :
                    (size_t)w_size * small_block_count + (size_t)(split_rank - s_bitmap[step]);
    s_offset = (s_bitmap[step] <= split_rank) ?
                (ptrdiff_t) s_bitmap[step] * (ptrdiff_t)(large_block_count * extent) :
                (ptrdiff_t)(s_bitmap[step] * (int) small_block_count + split_rank) * (ptrdiff_t) extent;

    r_count = (r_bitmap[step] + w_size <= split_rank) ?
                (size_t)w_size * large_block_count :
                  (r_bitmap[step] >= split_rank) ?
                    (size_t)w_size * small_block_count :
                    (size_t)w_size * small_block_count + (size_t)(split_rank - r_bitmap[step]);
    r_offset = (r_bitmap[step] <= split_rank) ?
                (ptrdiff_t) r_bitmap[step] * (ptrdiff_t)(large_block_count * extent) :
                (ptrdiff_t)(r_bitmap[step] * (int)small_block_count + split_rank) * (ptrdiff_t) extent;
    
    tmp_send = (char *)recv_buf + s_offset;
    tmp_recv = (char *)recv_buf + r_offset;

    ompi_coll_base_sendrecv(tmp_recv, r_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, tmp_send, s_count, dtype, vdest, MCA_COLL_BASE_TAG_ALLREDUCE, comm, MPI_STATUS_IGNORE, rank);
    
    w_size <<= 1;
  }

  free(tmp_buf_raw);

  return MPI_SUCCESS;
}


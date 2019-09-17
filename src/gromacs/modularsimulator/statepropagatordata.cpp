/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
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
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*! \libinternal
 * \brief Defines the state for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "statepropagatordata.h"

#include "gromacs/domdec/domdec.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/mdoutf.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/topology/atoms.h"

namespace gmx
{
StatePropagatorData::StatePropagatorData(
        int               numAtoms,
        FILE             *fplog,
        const t_commrec  *cr,
        t_state          *globalState,
        int               nstxout,
        int               nstvout,
        int               nstfout,
        int               nstxout_compressed,
        bool              useGPU,
        const t_inputrec *inputrec,
        const t_mdatoms  *mdatoms) :
    totalNumAtoms_(numAtoms),
    nstxout_(nstxout),
    nstvout_(nstvout),
    nstfout_(nstfout),
    nstxout_compressed_(nstxout_compressed),
    localNAtoms_(0),
    ddpCount_(0),
    writeOutStep_(-1),
    vvResetVelocities_(false),
    fplog_(fplog),
    cr_(cr),
    globalState_(globalState)
{
    // Initialize these here, as box_{{0}} in the initialization list
    // is confusing uncrustify and doxygen
    clear_mat(box_);
    clear_mat(previousBox_);

    bool stateHasVelocities;
    // Local state only becomes valid now.
    if (DOMAINDECOMP(cr))
    {
        auto localState = std::make_unique<t_state>();
        if (useGPU)
        {
            changePinningPolicy(&x_, gmx::PinningPolicy::PinnedIfSupported);
        }
        dd_init_local_state(cr->dd, globalState, localState.get());
        stateHasVelocities = static_cast<unsigned int>(localState->flags) & (1u << estV);
        setLocalState(std::move(localState));
    }
    else
    {
        state_change_natoms(globalState, globalState->natoms);
        f_.resizeWithPadding(globalState->natoms);
        localNAtoms_ = globalState->natoms;
        x_           = globalState->x;
        v_           = globalState->v;
        copy_mat(globalState->box, box_);
        stateHasVelocities = static_cast<unsigned int>(globalState->flags) & (1u << estV);
        previousX_.resizeWithPadding(localNAtoms_);
        ddpCount_ = globalState->ddp_count;
        copyPosition();
    }

    if (!inputrec->bContinuation)
    {
        if (stateHasVelocities)
        {
            auto v = velocitiesView().paddedArrayRef();
            // Set the velocities of vsites, shells and frozen atoms to zero
            for (int i = 0; i < mdatoms->homenr; i++)
            {
                if (mdatoms->ptype[i] == eptVSite ||
                    mdatoms->ptype[i] == eptShell)
                {
                    clear_rvec(v[i]);
                }
                else if (mdatoms->cFREEZE)
                {
                    for (int m = 0; m < DIM; m++)
                    {
                        if (inputrec->opts.nFreeze[mdatoms->cFREEZE[i]][m])
                        {
                            v[i][m] = 0;
                        }
                    }
                }
            }
        }
        if (inputrec->eI == eiVV)
        {
            vvResetVelocities_ = true;
        }
    }
}

ArrayRefWithPadding<RVec> StatePropagatorData::positionsView()
{
    return x_.arrayRefWithPadding();
}

ArrayRefWithPadding<const RVec> StatePropagatorData::constPositionsView() const
{
    return x_.constArrayRefWithPadding();
}

ArrayRefWithPadding<RVec> StatePropagatorData::previousPositionsView()
{
    return previousX_.arrayRefWithPadding();
}

ArrayRefWithPadding<const RVec> StatePropagatorData::constPreviousPositionsView() const
{
    return previousX_.constArrayRefWithPadding();
}

ArrayRefWithPadding<RVec> StatePropagatorData::velocitiesView()
{
    return v_.arrayRefWithPadding();
}

ArrayRefWithPadding<const RVec> StatePropagatorData::constVelocitiesView() const
{
    return v_.constArrayRefWithPadding();
}

ArrayRefWithPadding<RVec> StatePropagatorData::forcesView()
{
    return f_.arrayRefWithPadding();
}

ArrayRefWithPadding<const RVec> StatePropagatorData::constForcesView() const
{
    return f_.constArrayRefWithPadding();
}

rvec* StatePropagatorData::box()
{
    return box_;
}

const rvec* StatePropagatorData::constBox()
{
    return box_;
}

rvec* StatePropagatorData::previousBox()
{
    return previousBox_;
}

const rvec* StatePropagatorData::constPreviousBox()
{
    return previousBox_;
}

int StatePropagatorData::localNumAtoms()
{
    return localNAtoms_;
}

std::unique_ptr<t_state> StatePropagatorData::localState()
{
    auto state = std::make_unique<t_state>();
    state->flags = (1u << estX) | (1u << estV) | (1u << estBOX);
    state_change_natoms(state.get(), localNAtoms_);
    state->x = x_;
    state->v = v_;
    copy_mat(box_, state->box);
    state->ddp_count = ddpCount_;
    return state;
}

void StatePropagatorData::setLocalState(std::unique_ptr<t_state> state)
{
    localNAtoms_ = state->natoms;
    x_.resizeWithPadding(localNAtoms_);
    previousX_.resizeWithPadding(localNAtoms_);
    v_.resizeWithPadding(localNAtoms_);
    x_ = state->x;
    v_ = state->v;
    copy_mat(state->box, box_);
    copyPosition();
    ddpCount_ = state->ddp_count;
}

t_state* StatePropagatorData::globalState()
{
    return globalState_;
}

PaddedHostVector<RVec>* StatePropagatorData::forcePointer()
{
    return &f_;
}

void StatePropagatorData::copyPosition()
{
    int nth = gmx_omp_nthreads_get(emntUpdate);

    #pragma omp parallel for num_threads(nth) schedule(static) default(none) shared(nth)
    for (int th = 0; th < nth; th++)
    {
        int start_th, end_th;
        getThreadAtomRange(nth, th, localNAtoms_, &start_th, &end_th);
        copyPosition(start_th, end_th);
    }

    /* Box is changed in update() when we do pressure coupling,
     * but we should still use the old box for energy corrections and when
     * writing it to the energy file, so it matches the trajectory files for
     * the same timestep above. Make a copy in a separate array.
     */
    copy_mat(box_, previousBox_);
}

void StatePropagatorData::copyPosition(int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        previousX_[i] = x_[i];
    }
}

void StatePropagatorData::scheduleTask(
        Step step, Time gmx_unused time,
        const RegisterRunFunctionPtr &registerRunFunction)
{
    if (vvResetVelocities_)
    {
        vvResetVelocities_ = false;
        (*registerRunFunction)(
                std::make_unique<SimulatorRunFunction>(
                        [this](){resetVelocities(); }));
    }
    // copy x -> previousX
    (*registerRunFunction)(
            std::make_unique<SimulatorRunFunction>(
                    [this](){copyPosition(); }));
    // if it's a write out step, keep a copy for writeout
    if (step == writeOutStep_)
    {
        (*registerRunFunction)(
                std::make_unique<SimulatorRunFunction>(
                        [this](){saveState(); }));
    }
}

void StatePropagatorData::saveState()
{
    GMX_ASSERT(
            !localStateBackup_,
            "Save state called again before previous state was written.");
    localStateBackup_ = localState();
}

SignallerCallbackPtr
StatePropagatorData::registerTrajectorySignallerCallback(TrajectoryEvent event)
{
    if (event == TrajectoryEvent::stateWritingStep)
    {
        return std::make_unique<SignallerCallback>(
                [this](Step step, Time){this->writeOutStep_ = step; });
    }
    return nullptr;
}

ITrajectoryWriterCallbackPtr
StatePropagatorData::registerTrajectoryWriterCallback(TrajectoryEvent event)
{
    if (event == TrajectoryEvent::stateWritingStep)
    {
        return std::make_unique<ITrajectoryWriterCallback>(
                [this](gmx_mdoutf *outf, Step step, Time time)
                {write(outf, step, time); });
    }
    return nullptr;
}

void StatePropagatorData::write(gmx_mdoutf_t outf, Step currentStep, Time currentTime)
{
    unsigned int mdof_flags = 0;
    if (do_per_step(currentStep, nstxout_))
    {
        mdof_flags |= MDOF_X;
    }
    if (do_per_step(currentStep, nstvout_))
    {
        mdof_flags |= MDOF_V;
    }
    if (do_per_step(currentStep, nstfout_))
    {
        mdof_flags |= MDOF_F;
    }
    if (do_per_step(currentStep, nstxout_compressed_))
    {
        mdof_flags |= MDOF_X_COMPRESSED;
    }
    if (do_per_step(currentStep, mdoutf_get_tng_box_output_interval(outf)))
    {
        mdof_flags |= MDOF_BOX;
    }
    if (do_per_step(currentStep, mdoutf_get_tng_lambda_output_interval(outf)))
    {
        mdof_flags |= MDOF_LAMBDA;
    }
    if (do_per_step(currentStep, mdoutf_get_tng_compressed_box_output_interval(outf)))
    {
        mdof_flags |= MDOF_BOX_COMPRESSED;
    }
    if (do_per_step(currentStep, mdoutf_get_tng_compressed_lambda_output_interval(outf)))
    {
        mdof_flags |= MDOF_LAMBDA_COMPRESSED;
    }

    if (mdof_flags == 0)
    {
        return;
    }
    GMX_ASSERT(localStateBackup_, "Trajectory writing called, but no state saved.");

    // TODO: This is only used for CPT - needs to be filled when we turn CPT back on
    ObservablesHistory *observablesHistory = nullptr;

    mdoutf_write_to_trajectory_files(
            fplog_, cr_, outf, static_cast<int>(mdof_flags), totalNumAtoms_,
            currentStep, currentTime, localStateBackup_.get(), globalState_, observablesHistory, f_);

    localStateBackup_.reset();
}

void StatePropagatorData::elementSetup()
{
    if (vvResetVelocities_)
    {
        velocityBackup_ = v_;
    }
}

void StatePropagatorData::resetVelocities()
{
    v_ = velocityBackup_;
}

void StatePropagatorData::writeCheckpoint(t_state *localState, t_state gmx_unused *globalState)
{
    state_change_natoms(localState, localNAtoms_);
    localState->x = x_;
    localState->v = v_;
    copy_mat(box_, localState->box);
    localState->ddp_count = ddpCount_;
    localState->flags    |= (1u << estX) | (1u << estV) | (1u << estBOX);
}

}  // namespace gmx

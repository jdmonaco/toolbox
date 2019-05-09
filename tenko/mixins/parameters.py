"""
Simple swarmalator-like models of swarms and single agents.
"""

try:
    import simplejson as json
except ImportError:
    import json

import os
import subprocess
from collections import OrderedDict

from numpy import (ndarray, empty, empty_like, zeros, zeros_like, ones, eye,
        cos, sin, tanh, exp, expm1, log, log1p, sqrt, pi, newaxis, dot, diag,
        hypot, average, arange, broadcast_to, any)
from numpy.random import seed, rand, randn, randint
from numpy.ma import MaskedArray
TWOPI = 2*pi

from matplotlib.animation import FuncAnimation
from matplotlib.colors import colorConverter
from matplotlib import pyplot as plt, rcParams

from maps.geometry import EnvironmentGeometry, map_index, CUE_COLOR, REWARD_COLOR
from maps.types import *

from . import NeuroswarmsContext, step
from .matrix import *


class NeuroswarmModels(NeuroswarmsContext):

    def set_parameters(self, pfile=None, **params):
        """
        Set model parameters according to file, keywords, or defaults.
        """
        # Model parameters - Import from parameters file
        if pfile is not None:
            if not pfile.endswith('.json'):
                pfile += '.json'
            if os.path.isfile(pfile):
                pfilepath = pfile
            else:
                ctxpfile = os.path.join(self._ctxdir, pfile)
                if os.path.isfile(ctxpfile):
                    pfilepath = ctxpfile
                else:
                    self.out(pfile, prefix='MissingParamFile', error=True)
                    raise ValueError('Unable to find parameter file')
            with open(pfilepath, 'r') as fd:
                params_from_file = json.load(fd)
            params_from_file.update(params)
            params = params_from_file
            self.out(pfile, prefix='LoadedParameters')

        # Model parameters - Default values
        defaults = OrderedDict(
            model_type   = 'nsw',
            env          = 'test',
            show_rewards = False,
            show_cues    = False,
            test         = None,
            dt           = 0.01,
            duration     = 5.0,
            rnd_seed     = 'neuroswarmalators',
            N_S          = 300,
            single_agent = False,
            D_max        = 120.0,
            E_max        = 2.5e3,
            mu           = 0.9,
            m_mean       = 0.3,
            m_std        = 0.0,
            sigma        = 60.0,
            kappa        = 120.0,
            eta_S        = 0.01,
            eta_R        = 0.01,
            f            = 0.0,
            tau_p        = 0.1,
            tau_q        = 0.1,
            tau_c        = 0.1,
            tau_r        = 0.1,
            tau_u        = 0.1,
            lmbda        = 0.5,
            g_C          = 0.333,
            g_R          = 0.333,
            g_S          = 0.333,
            epsilon_C    = 0.0,
            epsilon_R    = 0.0,
            P_sf         = 0.0,
            sigma_rw     = 2.0,
            A_sw         = 1.0,
            B_sw         = 1.0,
            J_sw         = 1.0,
            K_sw         = 1.0
        )

        # Model parameter - Write JSON file with parameter defaults
        kwjson = dict(indent=2, separators=(', ', ': '))
        dfltpath = self.path('defaults.json')
        with open(dfltpath, 'w') as fd:
            json.dump({k:v for k,v in defaults.items() if v is not None},
                    fd, **kwjson)

        # Model parameter - Write JSON file with actual updated parameters
        psavefn = '' if self._tag is None else self._tag
        psavefn += '-{}'.format(self._lastcall['step'].replace('_', '-'))
        tag = self._lastcall['tag']
        if tag is not None:
            psavefn += '-{}'.format(tag.strip().lower().replace(' ', '-'))
        psavefn += '.json'
        parampath = self.path(psavefn)
        dparams = defaults.copy()
        dparams.update(params)
        with open(parampath, 'w') as fd:
            json.dump({k:v for k,v in dparams.items() if v is not None},
                    fd, **kwjson)

        # Force tag to test name if a test is being performed
        if 'test' in dparams and dparams['test'] is not None:
            self._lastcall['tag'] = params['test']
            self.out('\'{}\'', params['test'], prefix='RunningTest')

        # Model parameters - Set as global variables and log the values
        self.out('Independent parameters:')
        for name, dflt in defaults.items():
            exec(f'global {name}; {name} = dparams[\'{name}\']')
            val = globals()[name]
            if val == dflt:
                logstr = f' - {name} = {dflt}'
            else:
                logstr = f' * {name} = {val} [default: {dflt}]'
            self.out(logstr, hideprefix=True)

        # Environment - Import environmental geometry into the global scope,
        # into the persistent key-value store, and as instance attributes of
        # the simulation object
        global E
        self.hline()
        E = self.E = EnvironmentGeometry(env)
        Ivars = list(sorted(E.info.keys()))
        Avars = list(sorted(filter(lambda k: isinstance(getattr(E, k),
            ndarray), E.__dict__.keys())))
        self.out(repr(E), prefix='Geometry')
        for v in Ivars:
            exec(f'global {v}; self[\'{v}\'] = self.{v} = {v} = E.info[\'{v}\']')
            self.out(f' - {v} = {self[v]}', prefix='Geometry', hideprefix=True)
        for k in Avars:
            exec(f'global {k}; self.{k} = {k} = E.{k}')
            self.out(' - {} ({})', k, 'x'.join(list(map(str, getattr(self,
                k).shape))), prefix='Geometry', hideprefix=True)
        self['env'] = env
        self._save_env()

        # Other parameters - Dependent values
        self.hline()
        self.out('Dependent parameters:')
        depvalues = OrderedDict(
            N        = 1 if single_agent else N_S,
            Omega_0  = 2*pi*f,
            A_scaled = A_sw*G_scale,
            B_scaled = B_sw*G_scale**2,
            J_scaled = J_sw*G_scale,
            K_scaled = K_sw*G_scale
        )
        for name, val in depvalues.items():
            exec(f'global {name}; {name} = {val}')
            self.out(f' - {name} = {val}', hideprefix=True)
        self.hline()

    @step
    def simulate(self, tag=None, paramfile=None, **params):
        """
        Omnibus testing & development simulation. This will need to be
        refactored at some point to be sane.
        """
        # Basic simulation setup
        self.set_parameters(paramfile, **params)
        seed(sum(list(map(ord, rnd_seed))))
        if test is not None: tag = test

        # Model variables - Array allocation
        X         = empty((N, 2), dtype     = DISTANCE_DTYPE)
        X_hat     = empty((N, 2), dtype     = DISTANCE_DTYPE)
        X_S       = empty((N_S, 2), dtype   = DISTANCE_DTYPE)
        X_S_hat   = empty((N_S, 2), dtype   = DISTANCE_DTYPE)
        M         = empty((N, 1), dtype     = KILOGRAM_DTYPE)
        v_max     = empty((N, 1), dtype     = DISTANCE_DTYPE)
        v         = empty((N, 2), dtype     = DISTANCE_DTYPE)
        v_S       = empty((N, 2), dtype     = DISTANCE_DTYPE)
        v_m       = empty((N, 2), dtype     = DISTANCE_DTYPE)
        v_norm    = MaskedArray(empty((N, 2), dtype=DISTANCE_DTYPE))
        v_k       = empty((N, 2), dtype     = DISTANCE_DTYPE)
        b         = empty((N, 1), dtype     = DISTANCE_DTYPE)
        n_b       = empty((N, 2), dtype     = DISTANCE_DTYPE)
        b_S       = empty((N_S, 1), dtype   = DISTANCE_DTYPE)
        n_bS      = empty((N_S, 2), dtype   = DISTANCE_DTYPE)
        H         = empty((N, 1), dtype     = TILE_INDEX_DTYPE)
        H_S       = empty((N_S, 1), dtype   = TILE_INDEX_DTYPE)
        Delta     = MaskedArray(empty((N_S, 1), dtype=DISTANCE_DTYPE))
        D_S       = MaskedArray(empty((N_S, N_S), dtype=DISTANCE_DTYPE))
        dU        = MaskedArray(empty((N_S, N_S, 2), dtype=DISTANCE_DTYPE))
        D_S_up    = empty((N_S, N_S), dtype = DISTANCE_DTYPE)
        D_C       = MaskedArray(empty((N, N_C), dtype=DISTANCE_DTYPE))
        D_CS      = empty((N_S, N_C), dtype = DISTANCE_DTYPE)
        D_R       = MaskedArray(empty((N, N_R), dtype=DISTANCE_DTYPE))
        D_R_up    = empty((N_S, N_R), dtype = DISTANCE_DTYPE)
        D_RS      = empty((N_S, N_R), dtype = DISTANCE_DTYPE)
        D_RS_up   = empty((N_S, N_R), dtype = DISTANCE_DTYPE)
        V_DS      = empty((N_S, N_S), dtype = BOOL_DTYPE)
        V_GS      = empty((N_S, N_S), dtype = BOOL_DTYPE)
        V_mask    = empty((N_S, N_S), dtype = BOOL_DTYPE)
        V_mask    = empty((N_S, N_S), dtype = BOOL_DTYPE)
        W_S       = empty((N_S, N_S), dtype = WEIGHT_DTYPE)
        dX        = empty((N_S, 2), dtype   = DISTANCE_DTYPE)
        dX_norm   = empty((N_S, 2), dtype   = DISTANCE_DTYPE)
        dX_S      = empty((N_S, 2), dtype   = DISTANCE_DTYPE)
        W_R       = empty((N_S, N_R), dtype = WEIGHT_DTYPE)
        dX_R      = empty((N_S, 2), dtype   = DISTANCE_DTYPE)
        Theta     = empty((N_S, 1), dtype   = PHASE_DTYPE)
        dTheta    = MaskedArray(empty((N_S, N_S), dtype=PHASE_DTYPE))
        Omega_I   = empty((N_S, 1), dtype = PHASE_DTYPE)
        I_C       = empty((N, 1), dtype = WEIGHT_DTYPE)
        I_R       = empty((N, 1), dtype = WEIGHT_DTYPE)
        I_S       = empty((N, 1), dtype = WEIGHT_DTYPE)
        p         = empty((N_S, 1), dtype = WEIGHT_DTYPE)
        q         = empty((N_S, N_S), dtype = WEIGHT_DTYPE)
        c         = empty((N_S, 1), dtype = WEIGHT_DTYPE)
        r         = empty((N_S, N_R), dtype = WEIGHT_DTYPE)
        xi_q      = empty((N_S, N_S), dtype = WEIGHT_DTYPE)
        xi_c      = empty((N_S, N_C), dtype = WEIGHT_DTYPE)
        xi_r      = empty((N_S, N_R), dtype = WEIGHT_DTYPE)

        # Model variables - Initialization
        X[:]         = E.sample_spawn_points(N=N)
        X_hat[:]     = 0.0
        X_S[:]       = E.sample_spawn_points(N=N_S) if single_agent else X
        X_S_hat[:]   = 0.0
        M[:]         = m_mean + m_std*randn(N,1)
        v_max[:]     = sqrt((2*E_max) / M)
        v[:]         = 0.0
        v_S[:]       = 0.0
        v_m[:]       = 0.0
        v_norm[:]    = 0.0
        v_norm.mask  = broadcast_to(np.isclose(v_norm, 0), v_norm.shape)
        v_k[:]       = 0.0
        b[:]         = 0.0
        n_b[:]       = 0.0
        n_bS[:]      = 0.0
        H[:]         = G_PH[map_index(X)].data[:,newaxis]
        H_S[:]       = G_PH[map_index(X_S)].data[:,newaxis]
        Delta[:]     = distances(X, X_S)
        if single_agent:
            D_S[:]   = pairwise_distances(X_S, X_S)
        else:
            D_S[:]   = pairwise_distances(X, X)
        D_S_up[:]    = 0.0
        D_C[:]       = D_PC[map_index(X)]
        D_CS[:]      = D_PC[map_index(X_S)]
        D_R[:]       = D_PR[map_index(X)]
        D_C.mask[:]  = V_HC[H].squeeze()
        D_R.mask[:]  = V_HR[H].squeeze()
        D_R_up[:]    = 0.0
        D_RS[:]      = D_PR[map_index(X_S)]
        D_RS_up[:]   = 0.0
        if single_agent:
            V_DS[:]  = (D_S > 0.0)
            V_GS[:]  = V_HH[pairwise_tile_index(H_S, H_S)]
        else:
            V_DS[:]  = (D_S > 0.0) & (D_S <= D_max)
            V_GS[:]  = V_HH[pairwise_tile_index(H, H)]
        V_mask[:]    = ~(V_DS & V_GS)
        D_S.mask     = V_mask
        if single_agent:
            Delta.mask = ~V_HH[tile_index(H, H_S)]
            dU[:]    = -1*pairwise_unit_diffs(X_S, X_S, mask=V_mask)  # masked
        else:
            dU[:]    = -1*pairwise_unit_diffs(X, X, mask=V_mask)  # masked
        W_S[:]       = exp(-D_S/sigma)  # masked (via D_S)
        dX_S[:]      = 0.0
        dX[:]        = 0.0
        dX_norm[:] = 0.0
        W_R[:]       = rand(N_S, N_R)
        dX_R[:]      = 0.0
        Theta[:]     = TWOPI*rand(N_S, 1)
        dTheta[:]    = -1*pairwise_phasediffs(Theta, Theta, mask=V_mask)  # masked
        Omega_I[:]   = 0.0
        I_C[:]       = 0.0
        I_R[:]       = 0.0
        I_S[:]       = 0.0
        p[:]         = 0.0
        q[:]         = 0.0
        c[:]         = 0.0
        r[:]         = 0.0
        xi_q[:]      = randn(N_S, N_S)
        xi_c[:]      = randn(N_S, N_C)
        xi_r[:]      = randn(N_S, N_R)

        # Simulation time tracking
        _timesteps = arange(0, duration+dt, dt)
        _nframes   = len(_timesteps)

        # Position, velocity, and activation matrixes for recording
        _X         = zeros((_nframes,) + X.shape, X.dtype)
        _X_S       = zeros((_nframes,) + X_S.shape, X_S.dtype)
        _H_S       = zeros((_nframes,) + H_S.shape, H_S.dtype)
        _v         = zeros((_nframes,) + v.shape, v.dtype)
        _Theta     = zeros((_nframes,) + Theta.shape, Theta.dtype)
        _Omega     = zeros((_nframes,) + Omega_I.shape, Omega_I.dtype)

        # Plot formatting
        lw = 0.5
        agent_sz = 6
        cue_markers = 'v^<>12348sphHDd'
        reward_marker = '*'
        cr_fmt = dict(linewidths=lw, edgecolors='k', alpha=0.7, zorder=0)
        x_fmt = dict(marker='o', s=agent_sz, linewidths=lw, edgecolors='k',
                alpha=0.5, zorder=10)
        xs_fmt = x_fmt.copy()
        xs_fmt.update(marker='.', s=1, c='k', linewidths=0, alpha=1, zorder=5)
        text_xy = (0.5*width, 0.05*height)
        text_fmt = dict(va='top', ha='center', color='gray',
                fontsize='xx-small', fontweight='normal')

        # Colors and colormaps for visibility, distance, and tile tests
        swarm_colors = zeros((N_S, 4))
        vis_self = colorConverter.to_rgba('limegreen')
        vis_vis = colorConverter.to_rgba('red')
        vis_invis = colorConverter.to_rgba('dimgray')
        dist_cmap = plt.get_cmap('gnuplot2')
        tile_cmap = plt.get_cmap('tab20')
        X_cmap = plt.get_cmap('hsv')

        # Respawn positions that are transcending the matrix
        def respawn_oob_positions(X_pts):
            respawn  = X_pts[:,0] < 0
            respawn |= X_pts[:,1] < 0
            respawn |= X_pts[:,0] >= width
            respawn |= X_pts[:,1] >= height
            N_respawn = respawn.sum()
            if N_respawn:
                if self._out._hanging:
                    self.newline()
                self.out(f'{N_respawn} out-of-bound points',
                        prefix='Respawning', warning=True)
            X_pts[respawn] = E.sample_spawn_points(N=respawn.sum())
            return X_pts

        # Create figure window and init function for animation
        plt.ioff()
        _fig, _ax = E.figure(tag=tag, mapname='G_P', clear=True)
        self.figure(label='movie_frame', handle=_fig)
        self.artists = []
        self.cues = []
        def init():
            for i, _ in enumerate(C):
                self.cues.append(plt.scatter([], [], s=[],
                        marker=cue_markers[randint(len(cue_markers))],
                        c=CUE_COLOR, **cr_fmt))
            self.rewards = plt.scatter([], [], s=[], marker=reward_marker,
                    c=REWARD_COLOR, **cr_fmt)

            self._X_scatter = _ax.scatter([], [], cmap='hsv', vmin=0.0,
                    vmax=TWOPI, **x_fmt)
            self._X_S_scatter = _ax.scatter([], [], **xs_fmt)

            self._timestamp_text = _ax.text(text_xy[0], text_xy[1], '',
                    **text_fmt)

            self.artists.extend(self.cues)
            self.artists.extend([self.rewards, self._timestamp_text,
                self._X_scatter, self._X_S_scatter])

            return self.artists

        # Main loop - Update function
        def update(n):

            # Record current values of output variables
            t         = _timesteps[n]
            _X[n]     = X
            _X_S[n]   = X_S
            _H_S[n]   = H_S
            _v[n]     = v
            _Theta[n] = Theta
            _Omega[n] = Omega_I

            # Draw the cues and rewards on the first timestep
            if n == 0:
                if show_cues:
                    for i, xy in enumerate(C):
                        cue = self.cues[i]
                        cue.set_offsets(xy[newaxis])
                        cue.set_sizes([(k_H/3*(1+C_W[i]/3))**2])
                if show_rewards:
                    self.rewards.set_offsets(R)
                    self.rewards.set_sizes((k_H/3*(1+R_W/3))**2)

            # Update the progress bar and timestamp string
            if n % int(_nframes/100) == 0:
                self.box(filled=False, color='purple')
            self._timestamp_text.set_text(f't = {t:0.3f} s')

            # Update the scatter plots
            self._X_scatter.set_offsets(X)
            self._X_S_scatter.set_offsets(X_S)
            if test == 'vis':
                swarm_colors[V_GS[0].astype('?')] = vis_vis
                swarm_colors[~V_GS[0].astype('?')] = vis_invis
                swarm_colors[0] = vis_self
            elif test == 'delta':
                swarm_colors[:] = dist_cmap(np.clip(Delta.squeeze(), 0,
                    G_scale)/G_scale)
            elif test == 'dist':
                if n == 0:
                    swarm_colors[0] = vis_self
                swarm_colors[1:] = dist_cmap(np.clip(D_S[0,1:], 0,
                    G_scale)/G_scale)
            elif test == 'tile':
                swarm_colors[:] = tile_cmap((H_S if single_agent else H)/N_H)
            else:
                swarm_colors[:] = X_cmap(Theta.squeeze()/TWOPI)

            # Color the somatic points (single agent) or agent points (swarm)
            if single_agent:
                if n == 0:
                    self._X_scatter.set_facecolor('limegreen')
                self._X_S_scatter.set_color(swarm_colors)
            else:
                self._X_scatter.set_facecolor(swarm_colors)

            # Update the phase-space model: Random walkers (rw), Swarmalators
            # (sw), and NeuroSwarmalators (nsw).

            if model_type == 'rw':

                # Random-walk phase update
                Theta[:] = (Theta + 0.2*randn(N_S, 1)) % TWOPI

                # Random-walk of preferred locations that bounce off of walls
                X_S0 = X_S.copy()
                X_S0 += sigma_rw*randn(N_S, 2)
                while any(G_P[map_index(X_S0)]):
                    which = G_P[map_index(X_S0)].nonzero()[0]
                    X_S0[which] = X_S[which] + sigma_rw*randn(len(which), 2)
                dX_S[:] = X_S0 - X_S
                X_S[:] += dX_S

            elif model_type == 'sw':

                # Locally-windowed swarmalator attraction/repulsion
                dX[:] = dt*exp(-Delta**2/(2*sigma**2))*(
                    dU*(A_scaled + J_scaled*cos(dTheta[...,newaxis])) -
                    B_scaled*dU/D_S[...,newaxis]).mean(axis=1)

                # Swarmalator phase synchronization
                Theta[:] += dt*(Omega_0 +
                    K_scaled*(sin(dTheta)/D_S).mean(axis=1)[:,newaxis])
                Theta[:] %= TWOPI

                # Motion update: Geometric constraints for preferred locations
                X_S_hat[:] = respawn_oob_positions(X_S + dX)
                b_S[:] = G_PB[map_index(X_S_hat)][:,newaxis]
                n_bS[:] = G_PN[map_index(X_S_hat)]
                dX_norm[:] = hypot(dX[:,0], dX[:,1])[:,newaxis]
                dX[:] = (1-b_S)*dX + b_S*dX_norm*n_bS
                X_S[:] = respawn_oob_positions(X_S + dX)

            elif model_type == 'nsw':

                # Locally-windowed swarmalator attraction/repulsion
                # dX_S[:] = dt*exp(-Delta**2/(2*sigma**2))*(
                    # dU*(A_scaled + J_scaled*cos(dTheta[...,newaxis])) -
                    # B_scaled*dU/D_S[...,newaxis]).mean(axis=1)

                # Current updates
                c[:] = (dt/tau_c)*(exp(-(D_C-D_CS).sum(axis=1)**2 / (2*sigma**2))[:,newaxis] - c)
                r[:] = (dt/tau_r)*(exp(-D_R/kappa) - r)
                q[:] = (dt/tau_q)*Omega_I*(~V_mask).astype('i')

                # Conductance updates
                I_C[:] = g_C*c
                I_R[:] = g_R*(W_R*r).sum(axis=1)[:,newaxis]
                I_S[:] = g_S*(W_S*q).sum(axis=1)[:,newaxis]

                # Spike-phase updates
                Omega_I[:] = Omega_0*tanh((I_S + I_C + I_R) / Omega_0)
                Theta[:] += dt*(Omega_0 + Omega_I)
                Theta[:] %= TWOPI

                # Post-synaptic activity update
                p[:] = (dt/tau_p)*(Omega_I - p)

                # Weight-distance updates
                W_R[:] += eta_R*p*(r - p*W_R)
                W_S[:] += eta_S*p*(q - p*W_S)
                D_R_up[:] = -kappa*log(W_R)
                D_S_up[:] = -sigma*log(W_S)
                dX_R[:] = reward_position_update(D_R_up, D_R, X, R)
                dX_S[:] = somatic_position_update(D_S_up, D_S, X)

                dX[:] = (dX_S + dX_R)/2
                # X_S[:] = dX_S

                # Motion update: Geometric constraints for preferred locations
                X_S_hat[:] = respawn_oob_positions(X_S + dX)
                b_S[:] = G_PB[map_index(X_S_hat)][:,newaxis]
                n_bS[:] = G_PN[map_index(X_S_hat)]
                dX_norm[:] = hypot(dX[:,0], dX[:,1])[:,newaxis]
                dX[:] = (1-b_S)*dX + b_S*dX_norm*n_bS
                X_S[:] = respawn_oob_positions(X_S + dX)

            # Motion update: Somatic velocity guided by internal model
            if N == 1:
                # Single-agent case: The 'somata' velocity v_S is computed as
                # a weighted average of somata-agent position offsets, where the
                # weights are the magnitude of the shift in somata positions for
                # this update. The idea is that the strongest 'movement' in the
                # internal 'mental' model is what the agent should 'use' to make
                # its navigation decisions. That is, the largest changes in
                # place field locations draw the agent toward the updated
                # positions of those place fields.

                # Follow the resultant velocity flow. (Tends to accelerate into
                # walls; positive feedback b/n movement and swarm expansion.)
                v_S[:] = dX_S.sum(axis=0)/dt

                # Follow the normalized sum of the vector to the center of mass
                # of visible somata and the average positional shift
                # COM = (X_S[~Delta.mask.squeeze()].mean(axis=0) - X)
                # MOVE = dX_S.sum(axis=0)
                # COM_MOVE = COM + MOVE
                # NORM = hypot(COM_MOVE[:,0], COM_MOVE[:,1])
                # v_S[:] = COM_MOVE/(NORM*dt)

                # Trade-off random walking with flow following.
                # beta = (v_max - v_norm[:,0])/v_max
                # v_S[:] = (beta*dX_S.sum(axis=0) - (1-beta)*v_max*randn(1,2))/dt

            else:
                v_S[:] = (X_S - X)/dt

            # Motion update: Energetic constraints on momentum
            v_m[:]  = mu*v + (1-mu)*v_S
            v_norm[:] = hypot(v_m[:,0], v_m[:,1])[:,newaxis]
            v_norm.mask[:] = np.isclose(v_norm, 0)
            v_k[:]  = v_max*tanh(v_norm/v_max)*(v_m/v_norm)

            # Motion update: Geometric (barrier) constraints
            X_hat[:]  = X + v_k*dt
            b[:]      = G_PB[map_index(X_hat)][:,newaxis]
            n_b[:]    = G_PN[map_index(X_hat)]
            v[:]      = (1-b)*v_k + b*hypot(v_k[:,0], v_k[:,1])[:,newaxis]*n_b
            X[:]     += v*dt

            # Geometry update: Distances, visibility, and phase differences
            H[:]             = G_PH[map_index(X)].data[:,newaxis]
            H_S[:]           = G_PH[map_index(X_S)].data[:,newaxis]
            Delta[:]         = distances(X, X_S)
            if single_agent:
                D_S[:]       = pairwise_distances(X_S, X_S)
            else:
                D_S[:]       = pairwise_distances(X, X)
            D_C[:]           = D_PC[map_index(X)]
            D_C.mask[:]      = V_HC[H].squeeze()
            D_CS[:]          = D_PC[map_index(X_S)]
            D_R[:]           = D_PR[map_index(X)]
            D_R.mask[:]      = V_HR[H].squeeze()
            D_RS[:]          = D_PR[map_index(X_S)]
            if single_agent:
                V_DS[:]      = (D_S > 0.0)
                V_GS[:]      = V_HH[pairwise_tile_index(H_S, H_S)]
            else:
                V_DS[:]      = (D_S > 0.0) & (D_S <= D_max)
                V_GS[:]      = V_HH[pairwise_tile_index(H, H)]
            V_mask[:]        = ~(V_DS & V_GS)
            D_S.mask[:]      = V_mask
            if single_agent:
                Delta.mask[:] = ~V_HH[tile_index(H, H_S)]
                dU[:]        = -1*pairwise_unit_diffs(X_S, X_S, mask=V_mask)
            else:
                dU[:]        = -1*pairwise_unit_diffs(X, X, mask=V_mask)
            dTheta[:]        = -1*pairwise_phasediffs(Theta, Theta, mask=V_mask)

            # Reset weights to distance-based values
            W_S[:] = exp(-D_S/sigma)  # masked (via D_S)
            W_R[:] = exp(-D_R/kappa)  # masked (via D_R)
            D_S_up[:] = 0.0
            D_R_up[:] = 0.0

            # Reset intermediary differentials
            dX[:]      = 0.0
            dX_R[:]    = 0.0
            dX_S[:]    = 0.0
            dX_norm[:] = 0.0

            return self.artists

        # Create the animation object
        anim = FuncAnimation(fig=_fig, func=update, frames=range(_nframes),
                init_func=init, interval=10, repeat=False, blit=True)

        # Run the simulation and save the animation
        savefn = self._lastcall['step']
        if tag: savefn += '+{}'.format(self._norm_str(tag))
        savefn += '.mp4'
        self['moviefn'] = savefn
        temppath = self.path(savefn)
        anim.save(temppath, fps=100, dpi=245)
        self.hline()
        self.closefig()

        # Save simulation data to the datafile run root
        self.save_array(_timesteps, 't')
        self.save_array(_X, 'X')
        self.save_array(_X_S, 'X_S')
        self.save_array(_H_S, 'H_S')
        self.save_array(_v, 'v')
        self.save_array(_Theta, 'Theta')
        self.save_array(_Omega, 'Omega_I')

        # Play the movie if it was successfully saved
        if os.path.isfile(temppath):
            self.out(self._truncate(temppath), prefix='SavedMovie')
            self.play_movie(movie_path=temppath)

    def play_movie(self, movie_path=None):
        """
        Play the most recently saved movie.
        """
        if movie_path is not None:
            movp = movie_path
        elif 'moviefn' in self:
            movp = self.path(self.c.moviefn)
        else:
            self.out('Run the simulation or specify a path', error=True)
            return

        if not os.path.isfile(movp):
            self.out(self._truncate(movp), prefix='MissingFile', error=True)
            return

        dv = subprocess.DEVNULL
        devnull = dict(stdout=dv, stderr=dv)
        p = subprocess.run(['which', 'mpv'], **devnull)
        if p.returncode == 0:
            wscale = rcParams['figure.dpi']/self.c.G_scale
            subprocess.run(['mpv', '--loop=yes', '--ontop=yes',
                f'--window-scale={wscale:.1f}', movp], **devnull)
        else:
            self.out('Player \'mpv\' is missing', error=True)

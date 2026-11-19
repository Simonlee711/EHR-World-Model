"""
EHRWorldModel with an additive, state-dependent action term (``wm5plus``).

A predictor that is purely multiplicative in the latent velocity,

    z_{t+1} = z_t + F(z_t, a_t) dz_t,

can only re-scale / rotate the *current* displacement: when the patient is
near-stationary (dz_t ~ 0) no action can move the state.
``AdditiveActionEHRWorldModel`` adds an action-forcing term that vanishes exactly
when no action is given:

    z_{t+1} = z_t + F(z_t, a_t) dz_t + B(z_t, a_t),
    B(z, a) = h([z ; a]) - h([z ; 0])

Everything else (encoder, losses, SIGReg, reconstruction head, curriculum) is
inherited unchanged from ``ehr_world_model.models.EHRWorldModel``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ehr_world_model.models import EHRWorldModel


class AdditiveActionEHRWorldModel(EHRWorldModel):
    def __init__(self, *args, forcing_hidden: int = 128, forcing_scale: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.forcing_scale = forcing_scale       # ablation: fixed multiplier on B(z, a)
        D, A = self.latent_dim, self.n_actions
        self.forcing = nn.Sequential(nn.Linear(D + A, forcing_hidden), nn.SiLU(), nn.Linear(forcing_hidden, D))
        nn.init.zeros_(self.forcing[-1].weight)
        nn.init.zeros_(self.forcing[-1].bias)

    def action_forcing(self, z, action):
        return self.forcing_scale * (self.forcing(torch.cat([z, action], -1))
                                     - self.forcing(torch.cat([z, torch.zeros_like(action)], -1)))

    def _step(self, z, dz_prev, action):
        D = self.latent_dim
        F_m = self.predictor(torch.cat([z, action], -1)).view(-1, D, D)
        dz = torch.bmm(F_m, dz_prev.unsqueeze(-1)).squeeze(-1) + self.action_forcing(z, action)
        return torch.clamp(z + dz, -50.0, 50.0), dz

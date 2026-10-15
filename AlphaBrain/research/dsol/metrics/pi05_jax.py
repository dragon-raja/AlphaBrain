"""Opt-in OpenPI Pi0.5 velocity tracing without changing upstream files.

Adapted from OpenPI's Pi0.sample_actions (Apache-2.0). Preserve its while-loop
and cumulative time grid; record velocities before normalized action updates.
The caller must compare final actions against its installed upstream sampler.
"""


def sample_actions_with_trace(self, rng, observation, *, num_steps=10, noise=None):
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask

    if not isinstance(num_steps, int) or not 2 <= num_steps <= 100:
        raise ValueError('num_steps must be a static integer in [2, 100]')
    observation = model_api.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time, index, traces, times = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask,
            positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return (x_t + dt * velocity, time + dt, index + 1,
                traces.at[:, index].set(velocity), times.at[index].set(time))

    def cond(carry):
        return carry[1] >= -dt / 2

    trace = jnp.zeros((batch_size, num_steps, self.action_horizon, self.action_dim), dtype=noise.dtype)
    final, _, count, trace, times = jax.lax.while_loop(cond, step,
        (noise, 1.0, 0, trace, jnp.zeros((num_steps,), dtype=jnp.float32)))
    return dict(actions=final, velocity_trace=trace, times=times, step_count=count)


def make_velocity_sampler(model, *, num_steps=10):
    """Freeze model for inference; imports/JAX initialization only on explicit call."""
    from types import MethodType
    from openpi.shared.nnx_utils import module_jit
    compiled = module_jit(MethodType(sample_actions_with_trace, model), static_argnames=('num_steps',))

    def sample(rng, observation, *, noise):
        return compiled(rng, observation, noise=noise, num_steps=num_steps)

    return sample

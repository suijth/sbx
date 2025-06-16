from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Array, PRNGKey
from flax.linen import initializers, transforms
from flax.typing import Initializer
from jax import random

from flax.training.train_state import TrainState

from sbx.ppo.policies import PPOPolicy, Actor, Critic


class MultiLayerLSTMCell(nn.LSTMCell):
    num_layers: int = 1

    @nn.compact
    def __call__(self, carry, x):
        c, h = carry
        next_c, next_h = [], []
        for i in range(self.num_layers):
            n_carry, x = nn.LSTMCell(
                name=f"lstm_cell_{i}",
                features=self.features,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                recurrent_kernel_init=self.recurrent_kernel_init,
            )((c[i], h[i]), x)
            next_c.append(n_carry[0])
            next_h.append(n_carry[1])
        return (jnp.stack(next_c), jnp.stack(next_h)), x

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]) -> tuple[Array, Array]:
        c_list = []
        h_list = []
        for _ in range(self.num_layers):
            batch_dims = input_shape[:-1]
            key1, key2, rng = random.split(rng, 3)
            mem_shape = batch_dims + (self.features,)
            c_list.append(self.carry_init(key1, mem_shape, self.param_dtype))
            h_list.append(self.carry_init(key2, mem_shape, self.param_dtype))

        c = jnp.stack(c_list)
        h = jnp.stack(h_list)
        return (c, h)

class MultiLayerLSTMCellWithReset(MultiLayerLSTMCell):
    @nn.compact
    def __call__(self, carry, x, starts):
        print(carry[0].shape)
        if x.ndim > 1:
            carry = jax.vmap(lambda c,s: (c[0]* (1-s), c[1]*(1-s)), in_axes=((1, 1), 0),out_axes=(1,1))(carry, starts)
        else:
            carry = (carry[0]* (1-starts), carry[1]* (1-starts))
        return super().__call__(carry, x)

class ActorLSTM(Actor):
    lstm_hidden_size: int = 256
    num_layers: int = 1
    kernel_init: Initializer = None
    bias_init: Initializer = None
    recurrent_kernel_init: Initializer = None

    def setup(self):
        self.lstm_cell = MultiLayerLSTMCellWithReset(
            features=self.lstm_hidden_size,
            num_layers=self.num_layers,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            recurrent_kernel_init=self.recurrent_kernel_init
        )
        super().setup()

    @nn.compact
    def __call__(self, carry, x: jnp.ndarray, reset) -> jnp.ndarray:
        def apply_lstm(cell: MultiLayerLSTMCellWithReset, carry, x: jnp.ndarray, reset) -> jnp.ndarray:
            return cell(carry, x, reset)

        lstm = nn.scan(
                    apply_lstm,
                    variable_broadcast="params",
                    split_rngs={"params": False}, in_axes=0, out_axes=0
                )
        carry, x = lstm(self.lstm_cell, carry, x, reset)
        x = jnp.concatenate(x).transpose(0,1)
        return carry, super().__call__(x)

class CriticLSTM(Critic):
    lstm_hidden_size: int = 256
    lstm_cell: MultiLayerLSTMCellWithReset = None

    @nn.compact
    def __call__(self, carry, x: jnp.ndarray, reset) -> jnp.ndarray:
        def apply_lstm(cell: MultiLayerLSTMCellWithReset, carry, x: jnp.ndarray, reset) -> jnp.ndarray:
            return cell(carry, x, reset)

        lstm = nn.scan(
                    apply_lstm,
                    variable_broadcast="params",
                    split_rngs={"params": False}, in_axes=0, out_axes=0
                )
        carry, x = lstm(self.lstm_cell, carry, x, reset)
        return carry, super().__call__(x)


class RNNStatesNp(NamedTuple):
    pi: tuple[np.ndarray, ...]
    vf: tuple[np.ndarray, ...]

class RecurrentActorCriticPolicy(PPOPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.n_lstm_layers = n_lstm_layers
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

        assert not (
            self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the LSTM cannot be shared."

    def build(self, key: jax.Array, lr_schedule: Schedule, max_grad_norm: float) -> jax.Array:
        key, actor_key, vf_key = jax.random.split(key, 3)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs = {
                "action_dim": int(np.prod(self.action_space.shape)),
            }
        elif isinstance(self.action_space, spaces.Discrete):
            actor_kwargs = {
                "action_dim": int(self.action_space.n),
                "num_discrete_choices": int(self.action_space.n),
            }
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            assert self.action_space.nvec.ndim == 1, (
                f"Only one-dimensional MultiDiscrete action spaces are supported, "
                f"but found MultiDiscrete({(self.action_space.nvec).tolist()})."
            )
            actor_kwargs = {
                "action_dim": int(np.sum(self.action_space.nvec)),
                # type: ignore[dict-item]
                "num_discrete_choices": self.action_space.nvec,
            }
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            # Handle binary action spaces as discrete action spaces with two choices.
            actor_kwargs = {
                "action_dim": 2 * self.action_space.n,
                # type: ignore[dict-item]
                "num_discrete_choices": 2 * np.ones(self.action_space.n, dtype=int),
            }
        else:
            raise NotImplementedError(f"{self.action_space}")
        

        self.actor = ActorLSTM(
            lstm_hidden_size=self.lstm_output_dim,
            n_lstm_layers=self.n_lstm_layers,
            net_arch=self.net_arch_pi,
            # TODO: Add initializers
            **actor_kwargs,
        )

        obs = jnp.array([self.observation_space.sample()])
        resets = jnp.ones(obs.shape[:-1])
        carry = MultiLayerLSTMCellWithReset(
            features=self.lstm_output_dim,
            num_layers=self.n_lstm_layers,
        ).initialize_carry(
            key, obs.shape[1:]
        )
        #TODO: support shrared LSTM for the critic
        
        if self.enable_critic_lstm:
            self.critic = CriticLSTM(
                lstm_hidden_size=self.lstm_output_dim,
                n_lstm_layers=self.n_lstm_layers,
                net_arch=self.net_arch_vf,
                activation_fn=self.activation_fn,
            )
            params = self.critic.init(vf_key, carry, obs, resets)
        else:
            self.critic = Critic(
                net_arch=self.net_arch_vf,
                activation_fn=self.activation_fn,
            )
            params = self.critic.init(vf_key, obs)

        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # Inject hyperparameters to be able to modify it later
        # See https://stackoverflow.com/questions/78527164
        # Note: eps=1e-5 for Adam
        optimizer_class = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule(1), **self.optimizer_kwargs)

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, carry, jnp.array([obs]), jnp.array([resets])),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer_class,
            ),
        )

        self.vf_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer_class,
            ),
        )

        # type: ignore[method-assign]
        self.actor.apply = jax.jit(self.lstm_actor.apply)
        self.critic.apply = jax.jit(self.lstm_critic.apply)  # type: ignore[method-assign]

        return key

    def forward(
        self,
        obs: np.ndarray,
        lstm_states: RNNStatesNp,
        episode_starts: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, RNNStatesNp]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self.actor_state.apply_fn(self.actor_state.params, lstm_states.pi, obs, episode_starts)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (
                lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStatesNp(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[Distribution, tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(
            obs, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence(
            features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: np.ndarray,
        lstm_states: tuple[np.ndarray,np.ndarray],
        episode_starts: np.ndarray,
    ) -> np.ndarray:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(
                features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, lstm_states: RNNStates, episode_starts: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, lstm_states = self.get_distribution(
            observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate(
                [np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state[0], dtype=th.float32, device=self.device), th.tensor(
                state[1], dtype=th.float32, device=self.device
            )
            episode_starts = th.tensor(
                episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states

if __name__ == "__main__":
    def orthogonal(key):
        def init(key_given: Array,
                shape,
                dtype = jnp.float_) -> Array:
            return random.normal(key, shape, dtype)
        return init

    key = jax.random.key(0)

    ini = orthogonal(key)
    print(ini(key, (4, 256)))
    print(ini(key, (4, 256)))

    x = jnp.ones((5, 4))
    reset = jnp.ones((5,))
    lstm_layer = MultiLayerLSTMCellWithReset(features=256, 
                kernel_init=ini,
                bias_init=ini,
                recurrent_kernel_init=ini)
    # # vars1 = layer.init(key, x)
    # # c1, x = layer.apply(vars1,x, reset=jnp.ones(5))
    # lstm_cell = jax.jit(MultiLayerLSTMCellWithReset(features=256, num_layers=ini, has_batch_dim=x.ndim > 2,
    #                                 kernel_init=ini, bias_init=ini, recurrent_kernel_init=ini).apply)
    # net = nn.scan(
    #             lstm_cell,
    #             variable_broadcast="params",
    #             split_rngs={"params": False}, in_axes=0, out_axes=0
    #         )

    # carry = layer.initialize_carry(key, (4,))
    # vars1 = net.init(key, carry, x, reset)
    # ap = net.apply
    # c1 = ap(vars1, carry, x, reset)
    # print("done1")
    # c1 = ap(vars1, carry, x, reset)
    # print("done2")
    # x = jnp.ones((6, 4))
    # reset = jnp.ones((6,))
    # c1 = ap(vars1, carry, x, reset)
    # print(c1.shape)


    # x = jnp.ones((5, 4))
    # # layer = nn.LSTMCell(features=256, 
    # #             kernel_init=ini,
    # #             bias_init=ini,
    # #             recurrent_kernel_init=ini)
    # # net = nn.RNN(layer, return_carry=True)
    # # vars2 = net.init(key, x)
    # # out2, c2 = net.apply(vars2, x)
    # # print(out1[0].shape, out1[1].shape)
    layer = ActorLSTM(lstm_hidden_size=256, num_layers=1,
                kernel_init=ini, bias_init=ini, recurrent_kernel_init=ini, action_dim=10, net_arch=[])
    cell = MultiLayerLSTMCellWithReset(
            features=256,
            num_layers=1,
            kernel_init=ini,
            bias_init=ini,
            recurrent_kernel_init=ini
        )
    carry = cell.initialize_carry(key, (4,))
    # lstm_layer.init(key, carry, x[0], reset[0])
    vars2 = layer.init(key, carry, x, reset)
    c2 = layer.apply(vars2, carry, x, reset)
    c2 = layer.apply(vars2, carry, x, reset)
    # print(c2.shape)
    # print(c1-c2)
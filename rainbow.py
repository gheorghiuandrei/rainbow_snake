import argparse
import collections
import cv2
import numpy
import pygame
import pynput
import torch
import tqdm


class Network(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        frame_size = args.frame_size

        for kernel, stride in zip(args.kernels, args.strides):
            frame_size = (frame_size - kernel) // stride + 1

        channels = (args.state_size, *args.channels)
        conv_out_features = channels[-1] * frame_size ** 2
        self.actions = args.actions
        self.factor = args.factor
        self.bins = args.bins
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    args.kernels[i],
                    args.strides[i],
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.val1 = NoisyLinear(
            conv_out_features, args.hidden_features, args.sigma_zero
        )
        self.val2 = NoisyLinear(
            args.hidden_features, args.bins, args.sigma_zero
        )
        self.adv1 = NoisyLinear(
            conv_out_features, args.hidden_features, args.sigma_zero
        )
        self.adv2 = NoisyLinear(
            args.hidden_features, args.actions * args.bins, args.sigma_zero
        )

    def forward(self, x, log=False):
        for conv in self.convs:
            x = torch.nn.functional.relu(conv(x))

        x.register_hook(lambda x: x * self.factor)
        x = x.flatten(1)
        val = self.val2(torch.nn.functional.relu(self.val1(x)))
        val = val.view(-1, 1, self.bins)
        adv = self.adv2(torch.nn.functional.relu(self.adv1(x)))
        adv = adv.view(-1, self.actions, self.bins)
        x = val + adv - adv.mean(1, keepdim=True)

        if log:
            x = torch.nn.functional.log_softmax(x, 2)
        else:
            x = torch.nn.functional.softmax(x, 2)

        return x


class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma_zero):
        super().__init__()

        weight_size = (out_features, in_features)
        bias_size = out_features
        mu_bound = 1 / in_features ** 0.5
        sigma_constant = sigma_zero / in_features ** 0.5
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = torch.nn.Parameter(torch.empty(weight_size))
        self.mu_bias = torch.nn.Parameter(torch.empty(bias_size))
        self.sigma_weight = torch.nn.Parameter(torch.empty(weight_size))
        self.sigma_bias = torch.nn.Parameter(torch.empty(bias_size))
        torch.nn.init.uniform_(self.mu_weight, -mu_bound, mu_bound)
        torch.nn.init.uniform_(self.mu_bias, -mu_bound, mu_bound)
        torch.nn.init.constant_(self.sigma_weight, sigma_constant)
        torch.nn.init.constant_(self.sigma_bias, sigma_constant)

    def forward(self, x):
        epsilon_in = torch.randn(self.in_features).to(x.device)
        epsilon_out = torch.randn(self.out_features).to(x.device)
        epsilon_in = epsilon_in.sign() * epsilon_in.abs().sqrt()
        epsilon_out = epsilon_out.sign() * epsilon_out.abs().sqrt()
        epsilon_weight = epsilon_out.ger(epsilon_in)
        epsilon_bias = epsilon_out
        weight = self.mu_weight + self.sigma_weight * epsilon_weight
        bias = self.mu_bias + self.sigma_bias * epsilon_bias
        x = torch.nn.functional.linear(x, weight, bias)

        return x


class Memory:
    def __init__(self, args):
        self.discount = args.discount
        self.memory_size = args.memory_size
        self.index = 0
        self.exponent = args.exponent
        self.correction = args.correction
        self.correction_increase = (1 - args.correction) / (
            args.train_steps / args.replay_frequency
        )
        self.n = args.n
        self.transitions = []
        self.priorities = SumTree(args.memory_size)
        self.priorities_power = SumTree(args.memory_size)
        self.priorities_array = numpy.ones(args.memory_size)
        self.weights = numpy.full(args.memory_size, -numpy.inf)
        self.n_step_buffer = collections.deque()

    def append(self, transition):
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n:
            return

        *_, next_state, discount = self.n_step_buffer[-1]

        for _ in range(self.n if discount == 0 else 1):
            n_step_return = 0

            for i, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
                n_step_return += self.discount ** i * reward

            state, action, *_ = self.n_step_buffer.popleft()
            transition = (state, action, n_step_return, next_state, discount)

            if len(self.transitions) < self.memory_size:
                self.transitions.append(transition)
            else:
                self.transitions[self.index] = transition

            max_priority = self.priorities_array.max()
            self.priorities[self.index] = max_priority
            self.priorities_power[self.index] = max_priority ** self.exponent
            self.index = (self.index + 1) % self.memory_size

    def sample(self, batch_size):
        indices = []
        weights = []
        transitions = []
        segment = self.priorities.sum / batch_size

        for i in range(batch_size):
            priority = numpy.random.uniform(i * segment, (i + 1) * segment)
            index = self.priorities.get_index(priority)
            probability = (
                self.priorities_power[index] / self.priorities_power.sum
            )
            weight = (len(self.transitions) * probability) ** -self.correction
            self.weights[index] = weight
            indices.append(index)
            weights.append(weight)
            transitions.append(self.transitions[index])

        max_weight = self.weights.max()
        weights = [weight / max_weight for weight in weights]
        self.correction += self.correction_increase

        return indices, weights, transitions

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[index] = priority
            self.priorities_power[index] = priority ** self.exponent
            self.priorities_array[index] = priority


class SumTree:
    def __init__(self, size):
        self.tree_size = 2 ** int(numpy.ceil(numpy.log2(size)))
        self.tree = [0] * (2 * self.tree_size - 1)

    def __setitem__(self, index, value):
        index = index + self.tree_size - 1
        self.tree[index] = value
        self._update(index)

    def __getitem__(self, index):
        return self.tree[index + self.tree_size - 1]

    def _update(self, index):
        parent = (index - 1) // 2
        left = 2 * parent + 1
        right = 2 * parent + 2
        self.tree[parent] = self.tree[left] + self.tree[right]

        if parent != 0:
            self._update(parent)

    def get_index(self, value, index=0):
        left = 2 * index + 1
        right = 2 * index + 2

        if index >= self.tree_size - 1:
            return index - self.tree_size + 1
        elif value < self.tree[left]:
            return self.get_index(value, left)
        else:
            return self.get_index(value - self.tree[left], right)

    @property
    def sum(self):
        return self.tree[0]


class Agent:
    def __init__(self, args):
        self.online = Network(args).to(args.device)
        self.target = Network(args).to(args.device)
        self.update_target()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), args.learning_rate, eps=args.epsilon
        )
        self.memory = Memory(args)
        self.batch_size = args.batch_size
        self.n = args.n
        self.max_norm = args.max_norm
        self.v_min = -args.v_bound
        self.v_max = args.v_bound
        self.z = torch.linspace(self.v_min, self.v_max, args.bins)
        self.z = self.z.to(args.device)
        self.delta_z = (self.v_max - self.v_min) / (args.bins - 1)
        self.offset = torch.arange(0, args.batch_size * args.bins, args.bins)
        self.offset = self.offset.unsqueeze(1).repeat(1, args.bins)
        self.offset = self.offset.to(args.device)
        self.half = torch.full((args.batch_size, args.bins), 0.5)
        self.half = self.half.to(args.device)
        self.device = args.device

    def act(self, state):
        state = self._to_tensor(numpy.array(state)).unsqueeze(0) / 255
        action = (self.z * self.online(state)).sum(2).argmax(1).item()

        return action

    def replay(self):
        indices, weights, transitions = self.memory.sample(self.batch_size)
        states, actions, returns, next_states, discounts = zip(*transitions)
        weights = self._to_tensor(weights)
        states = self._to_tensor(numpy.array(states)) / 255
        returns = self._to_tensor(returns).unsqueeze(1)
        next_states = self._to_tensor(numpy.array(next_states)) / 255
        discounts = self._to_tensor(discounts).unsqueeze(1)
        next_actions = (self.z * self.online(next_states)).sum(2).argmax(1)
        next_p = self.target(next_states)[range(self.batch_size), next_actions]
        t_z = returns + discounts ** self.n * self.z
        t_z = t_z.clamp(self.v_min, self.v_max)
        b = (t_z - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        m = torch.zeros_like(b)
        m.put_(
            l + self.offset,
            next_p * torch.where(l != u, u - b, self.half),
            True,
        )
        m.put_(
            u + self.offset,
            next_p * torch.where(l != u, b - l, self.half),
            True,
        )
        log_p = self.online(states, log=True)[range(self.batch_size), actions]
        loss = weights * -(m * log_p).sum(1)
        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.max_norm)
        self.optimizer.step()
        self.memory.update_priorities(indices, loss.detach().cpu().numpy())

    def _to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float, device=self.device)

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, name):
        torch.save(self.online.state_dict(), name)

    def load(self, name):
        self.online.load_state_dict(torch.load(name, self.device))


class Environment:
    def __init__(self, args):
        self.environment = getattr(__import__(args.game), "Environment")()
        self.max_frames_per_episode = args.max_frames_per_episode
        self.frame_sizes = (args.frame_size,) * 2
        self.state_size = args.state_size
        self.frames_per_episode = None
        self.state = None
        self.done = False
        self.listener = pynput.keyboard.Listener(on_press=self._stop)

    def _stop(self, key):
        if key == pynput.keyboard.Key.esc:
            self.done = True

    def reset(self):
        self.frames_per_episode = 1
        frame = self.environment.reset()
        frame = self._preprocess(frame)
        self.state = (frame,) * self.state_size

        return self.state

    def step(self, action):
        self.frames_per_episode += 1
        reward, frame, terminal = self.environment.step(action)
        frame = self._preprocess(frame)
        self.state = (*self.state[1:], frame)

        if self.frames_per_episode == self.max_frames_per_episode:
            terminal = True

        return reward, self.state, terminal

    def _preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, self.frame_sizes)

        return frame

    @property
    def actions(self):
        return len(self.environment.actions)


def train(environment, agent, args):
    total_steps = args.observe_steps + args.train_steps
    terminal = True

    for t in tqdm.trange(1, total_steps + 1, desc="Train", leave=False):
        if terminal:
            state = environment.reset()

        action = agent.act(state)
        reward, next_state, terminal = environment.step(action)
        reward = numpy.clip(reward, -args.reward_bound, args.reward_bound)
        discount = args.discount * (1 - terminal)
        agent.memory.append((state, action, reward, next_state, discount))
        state = next_state

        if t <= args.observe_steps:
            continue

        if (t - args.observe_steps) % args.replay_frequency == 0:
            agent.replay()

        if (t - args.observe_steps) % args.update_target_frequency == 0:
            agent.update_target()

        if (t - args.observe_steps) % args.evaluate_frequency == 0:
            evaluate(agent, args)

    if args.save:
        agent.save(f"{args.game}.pt")


def evaluate(agent, args):
    environment = Environment(args)
    terminal = True
    scores = []

    for _ in tqdm.trange(1, args.evaluate_steps + 1, desc="Test", leave=False):
        if terminal:
            state = environment.reset()
            score = 0

        reward, state, terminal = environment.step(agent.act(state))
        score += reward

        if terminal:
            scores.append(score)

    with open("scores.txt", "a") as file:
        print(numpy.mean(scores).round(1), file=file)


def test(agent, args):
    agent.load(f"{args.game}.pt")
    environment = Environment(args)
    environment.listener.start()
    terminal = True

    while not environment.done:
        if terminal:
            state = environment.reset()

        _, state, terminal = environment.step(agent.act(state))
        pygame.time.delay(args.wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="snake")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    args.max_frames_per_episode = 9_000
    args.frame_size = 84
    args.state_size = 4
    args.observe_steps = 20_000
    args.train_steps = 50_000_000
    args.evaluate_steps = 125_000
    args.replay_frequency = 4
    args.update_target_frequency = 8_000
    args.evaluate_frequency = 1_000_000
    args.reward_bound = 1
    args.discount = 0.99
    args.channels = (32, 64, 64)
    args.kernels = (8, 4, 3)
    args.strides = (4, 2, 1)
    args.hidden_features = 512
    args.memory_size = 1_000_000
    args.exponent = 0.5
    args.correction = 0.4
    args.n = 3
    args.learning_rate = 0.0000625
    args.epsilon = 0.00015
    args.batch_size = 32
    args.factor = 1 / 2 ** 0.5
    args.max_norm = 10
    args.bins = 51
    args.v_bound = 10
    args.sigma_zero = 0.1
    args.wait = 100
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    environment = Environment(args)
    args.actions = environment.actions
    agent = Agent(args)
    train(environment, agent, args) if not args.test else test(agent, args)


if __name__ == "__main__":
    main()

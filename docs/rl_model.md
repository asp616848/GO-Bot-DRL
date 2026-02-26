# RL Model — GO-Bot-DRL

A complete reference for the Markov Decision Process formulation, state/action/reward design, neural network, and training procedure of the DQN-based dialogue agent.

---

## 1. Problem as an MDP

The movie-booking dialogue is framed as an episodic MDP:

$$\mathcal{M} = \langle \mathcal{S},\ \mathcal{A},\ \mathcal{T},\ \mathcal{R},\ \gamma \rangle$$

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | Continuous state space — 224-dim float vector per turn |
| $\mathcal{A}$ | Discrete action space — 40 possible dialogue actions |
| $\mathcal{T}$ | Stochastic transition — user simulator + error model |
| $\mathcal{R}$ | Scalar reward — see §3 |
| $\gamma$ | Discount factor = **0.9** |

An **episode** = one full conversation. It ends when the agent issues `done` or the turn limit (20) is hit.

---

## 2. State Space

`StateTracker.get_state()` assembles a single flat numpy vector of size:

$$|\mathcal{S}| = 2 \cdot |\text{intents}| + 7 \cdot |\text{slots}| + 3 + T_{\max} = 2(6) + 7(27) + 3 + 20 = \mathbf{224}$$

### Component Breakdown

| Segment | Size | Description |
|---|---|---|
| `user_act` | 6 | One-hot of the **current user intent** |
| `user_inform_slots` | 27 | Bag-of-slots: which slots the user *informed* this turn |
| `user_request_slots` | 27 | Bag-of-slots: which slots the user *requested* this turn |
| `agent_act` | 6 | One-hot of the **last agent intent** |
| `agent_inform_slots` | 27 | Bag-of-slots: which slots the agent informed last turn |
| `agent_request_slots` | 27 | Bag-of-slots: which slots the agent requested last turn |
| `current_slots` | 27 | Bag-of-slots: *all* accumulated inform slots so far |
| `turn_value` | 1 | Scalar turn number / 5 |
| `turn_onehot` | 20 | One-hot position in the episode |
| `kb_binary` | 28 | Per-slot: is there ≥ 1 DB match for that slot? (+ overall) |
| `kb_count` | 28 | Per-slot: fraction of DB matches for that slot (÷ 100) |

### State Vector Layout

```
 ◄──6──►◄──────27──────►◄──────27──────►◄──6──►◄──────27──────►◄──────27──────►◄──────27──────►◄1►◄──20──►◄──28──►◄──28──►
[usr_act][usr_inform    ][usr_request   ][agt_act][agt_inform   ][agt_request   ][cur_slots      ][t][turn_1h][kb_bin ][kb_cnt ]
 0     5  6           32  33          59  60   65  66         92  93        119  120       146  147 148   167 168  195 196  223
```

The KB (knowledge-base) section effectively tells the agent how many movies still match the accumulated constraints, which directly guides its information-gathering strategy.

---

## 3. Action Space

40 discrete actions, built at import time in `dialogue_config.py`:

```
Index 0:      { intent: 'done',        inform: {},          request: {} }
Index 1:      { intent: 'match_found', inform: {},          request: {} }
Index 2–20:   { intent: 'inform',      inform: {slot: PH},  request: {} }   — one per inform slot
Index 21–39:  { intent: 'request',     inform: {},          request: {slot: UNK} } — one per request slot
```

*`PH` = `'PLACEHOLDER'` — filled with a real DB value by `StateTracker.update_state_agent()`.*

### Action Semantics

| Intent | Meaning |
|---|---|
| `request` | Ask the user for a specific slot value |
| `inform` | Tell the user the value of a slot (pulled from DB) |
| `match_found` | Present a movie match; `inform_slots` filled with all match fields |
| `done` | End the conversation; triggers success/failure evaluation |

---

## 4. Reward Function

Non-zero rewards only occur at terminal steps. The per-turn "alive" cost discourages unnecessary turns.

$$r_t = \begin{cases} -1 + 2 \cdot T_{\max} = +39 & \text{SUCCESS} \\ -1 - T_{\max} = -21 & \text{FAIL} \\ -1 & \text{ongoing} \end{cases}$$

with $T_{\max} = 20$.

The large asymmetry (+39 vs −21 vs −1) strongly rewards efficient success and penalises failure over simply taking many steps.

**Success condition** (checked in `user_simulator._response_to_done()`):
1. The agent previously issued `match_found` AND all goal constraints matched.
2. The user's `rest_slots` are fully satisfied (nothing left to ask for).

---

## 5. DQN Agent

The agent approximates the action-value function $Q(s, a)$ with a two-layer MLP.

### Architecture

```
Input (224)
    │
    ▼
Dense(80, ReLU)    ← hidden_size = 80
    │
    ▼
Dense(40, Linear)  ← one output neuron per action
    │
    ▼
Q(s, a)  for all a ∈ A simultaneously
```

Two identical copies of this network are maintained:

| Network | Variable | Role |
|---|---|---|
| **Behaviour network** | `beh_model` | Selects actions; trained every `train_freq` episodes |
| **Target network** | `tar_model` | Provides stable TD targets; weights copied from `beh_model` |

### DQN vs DDQN

The `vanilla` flag (default `True`) switches the Bellman target:

**Vanilla DQN** (`vanilla=True`):
$$y = r + \gamma \cdot \max_{a'} Q_{\text{tar}}(s', a') \cdot \mathbb{1}[\neg \text{done}]$$

**Double DQN** (`vanilla=False`):
$$y = r + \gamma \cdot Q_{\text{tar}}\!\left(s',\, \arg\max_{a'} Q_{\text{beh}}(s', a')\right) \cdot \mathbb{1}[\neg \text{done}]$$

DDQN decorrelates action selection from value estimation, reducing overestimation bias.

---

## 6. Exploration — ε-Greedy

$$\pi(s) = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a Q_{\text{beh}}(s, a) & \text{with probability } 1 - \varepsilon \end{cases}$$

| Parameter | Default |
|---|---|
| `epsilon_init` | `0.0` (inference / fine-tune) |

*During training, ε is typically set > 0 to enable exploration. At inference/evaluation, ε = 0 for pure exploitation.*

During the **warmup phase**, the agent uses a hand-crafted **rule-based policy** instead of ε-greedy:

```
Rule policy request order: moviename → starttime → city → date → theater → numberofpeople
After all requests:        match_found  →  done
```

---

## 7. Experience Replay

Each transition $(s, a, r, s', \text{done})$ is stored in a circular replay buffer.

| Parameter | Value |
|---|---|
| `max_mem_size` | 500 000 |
| `batch_size` | 16 |

At each training step, `batch_size` transitions are sampled **uniformly** at random, breaking temporal correlations. The behavior model is then updated via a single epoch of `model.fit()`.

---

## 8. Training Loop

```
WARMUP PHASE
  loop until memory has WARMUP_MEM (1 000) transitions:
      reset episode
      run rounds using rule-based policy
      store (s, a, r, s', done) in memory

TRAINING PHASE  (40 000 episodes)
  for episode in range(NUM_EP_TRAIN):
      reset episode
      loop until done:
          s  ← state_tracker.get_state()
          a  ← agent.get_action(s)           # ε-greedy / DQN
          update state_tracker with a
          u, r, done, success ← user_sim.step(a)
          infuse_error(u)                     # EMC
          update state_tracker with u
          s' ← state_tracker.get_state(done)
          store (s, a, r, s', done)

      every TRAIN_FREQ (100) episodes:
          success_rate = successes / TRAIN_FREQ
          if success_rate >= best AND >= THRESHOLD (0.3):
              empty replay memory          # "memory flush"
          if success_rate > best:
              save weights
              best = success_rate
          train() — sample batch, compute TD targets, update beh_model

  copy beh_model → tar_model  (after each training block)
```

### Memory Flush

When a new best success rate above the threshold is achieved, the replay buffer is **emptied**. This prevents the agent from being dragged back towards old, suboptimal behaviour patterns as it improves.

---

## 9. Episode Lifecycle Diagram

```
                ┌─────────────────────────────────────────────────┐
                │               Episode Reset                      │
                │  state_tracker.reset()                           │
                │  user_sim.reset()  → initial user action         │
                │  dqn_agent.reset() → clear rule-based counters   │
                └───────────────────┬─────────────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  state = get_state() │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │  Agent: get_action(state)       │
                    │   ε-greedy or argmax Q(s,·)     │
                    └───────────────┬────────────────┘
                                    │ agent_action
                    ┌───────────────▼────────────────┐
                    │  StateTracker:                  │
                    │   fill PLACEHOLDER via DB query │
                    │   append to history             │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │  User Sim: step(agent_action)  │
                    │   → user_action, reward,        │
                    │     done, success               │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │  EMC: infuse_error(user_action) │
                    │  (skipped if done=True)         │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │  StateTracker:                  │
                    │   update current_informs        │
                    │   append user_action to history │
                    └───────────────┬────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │  next_state = get_state(done)   │
                    │  store (s,a,r,s',done)          │
                    └───────────────┬────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  done = False?       │──── Yes ──► (loop back)
                         └──────────┬──────────┘
                                    │ No
                         ┌──────────▼──────────┐
                         │  End of episode      │
                         │  Accumulate reward   │
                         └─────────────────────┘
```

---

## 10. Hyperparameter Summary

| Parameter | Value | Notes |
|---|---|---|
| `gamma` | 0.9 | Discount factor |
| `learning_rate` | 1e-3 | Adam optimiser |
| `hidden_size` | 80 | Single hidden layer neurons |
| `batch_size` | 16 | Replay sample size |
| `max_mem_size` | 500 000 | Circular replay buffer capacity |
| `epsilon_init` | 0.0 | Set > 0 during training for exploration |
| `warmup_mem` | 1 000 | Transitions filled by rule policy before training |
| `num_ep_run` | 40 000 | Total training episodes |
| `train_freq` | 100 | Episodes between gradient updates |
| `max_round_num` | 20 | Max turns per episode |
| `success_rate_threshold` | 0.3 | Minimum rate to trigger memory flush & weight save |
| `vanilla` | `true` | `true` = DQN, `false` = DDQN |
| `slot_error_prob` | 0.05 | Per-slot NLU noise probability |

---

## 11. Q-Value Interpretation

At any point during interaction you can inspect $Q(s, \cdot)$ to understand the agent's decision:

- High $Q$ on `request/moviename` early → agent hasn't learned the primary constraint yet  
- High $Q$ on `match_found` → current DB constraints narrow to a confident match  
- High $Q$ on `done` → agent believes the conversation goal has been satisfied  

The gap between the top-1 and top-2 Q-values is a rough proxy for the agent's confidence.

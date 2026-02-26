# Simulation & Data — GO-Bot-DRL

A complete reference for the data pipeline, dialogue configuration, user simulation, and error injection used to train the agent.

---

## 1. Data Files

Three pickle files live in `data/` and are loaded at startup:

| File | Variable | Contents |
|---|---|---|
| `movie_db.pkl` | `database` | `dict[int → dict]` — every movie entry with slot–value pairs |
| `movie_dict.pkl` | `db_dict` | `dict[slot → list[values]]` — all possible values per slot (used by EMC) |
| `movie_user_goals.pkl` | `user_goals` | `list[dict]` — pre-generated user goal objects |

The raw text equivalents (`movie_db.txt`, `movie_dict.txt`, `movie_user_goals.txt`) live alongside them; `pickle_converter.py` converts them. Because they were pickled with Python 2, loading requires `encoding='latin1'`.

### Database Entry Shape

```python
{
  391: {
    'moviename': 'zootopia',
    'theater':   'amc sunset 5',
    'starttime': '10:00 am',
    'date':      'tuesday',
    'city':      'seattle',
    'genre':     'animation',
    'critic_rating': 'good',
    'mpaa_rating':   'pg',
    'price':     '12',
    # ... other slots may be absent
  }
}
```

Empty-string values are stripped by `remove_empty_slots()` so the agent never sees them.

---

## 2. Dialogue Configuration (`dialogue_config.py`)

Everything that defines the vocabulary of the conversation lives here.

### Intents

| Party | Intents |
|---|---|
| **User / Simulator** | `inform`, `request`, `thanks`, `reject`, `done` |
| **Agent** | `inform`, `request`, `match_found`, `done` |
| **All (state encoding)** | `inform`, `request`, `done`, `match_found`, `thanks`, `reject` |

### Slots

27 named slots are recognised everywhere. Key ones:

| Slot | Role |
|---|---|
| `moviename` | Primary constraint; required in first user turn if in goal |
| `ticket` | The **default key** — the agent must inform this to signal a booking match |
| `numberofpeople`, `ticket` | **Non-queryable** — never passed to DB query |
| All others | Used freely in informs and requests |

### Agent Action Space

The full set of actions the agent can output is built programmatically:

```
done                                            (1 action)
match_found                                     (1 action)
inform / <slot>: PLACEHOLDER   for each inform slot except 'ticket'   (19 actions)
request / <slot>: UNK          for each request slot                   (19 actions)
─────────────────────────────────────────────────────────────
Total: 40 actions
```

`PLACEHOLDER` values are filled in at runtime by `StateTracker` via a DB query.

---

## 3. User Goals

Each goal is a dict with two sub-dicts:

```python
{
  'inform_slots':  {'moviename': 'zootopia', 'date': 'tuesday'},
  'request_slots': {'starttime': 'UNK', 'theater': 'UNK'}
}
```

* `inform_slots` — constraints the user *has* and will tell the agent.
* `request_slots` — information the user *wants* from the agent.
* The `ticket` key is **always** injected into `request_slots` at episode reset (the simulated user always wants a booking confirmation).

---

## 4. User Simulator (`user_simulator.py`)

The `UserSimulator` acts as the environment. It has an internal state machine that responds to agent actions deterministically (with mild stochasticity in slot ordering).

### State Variables

| Variable | Purpose |
|---|---|
| `goal` | Randomly sampled from `user_goals` at episode start |
| `state['history_slots']` | Slot–value pairs already communicated (by either side) |
| `state['rest_slots']` | Goal slots not yet fulfilled |
| `state['inform_slots']` | Inform payload for the *current* response |
| `state['request_slots']` | Request payload for the *current* response |
| `constraint_check` | `SUCCESS` once `match_found` is validated; `FAIL` otherwise |

### Episode Reset Flow

```
reset()
  ├─ Sample goal randomly
  ├─ Inject 'ticket' into goal request_slots
  ├─ Initialise history/rest/request/inform to empty
  └─ _return_init_action()
       ├─ intent = 'request'
       ├─ Add required init informs (moviename if in goal)
       └─ Add one random request slot
```

### Response Rules (`step()`)

The user sim receives the agent's action and routes to one of four handlers:

```
agent intent
  ├── 'request'      → _response_to_request()
  ├── 'inform'       → _response_to_inform()
  ├── 'match_found'  → _response_to_match_found()
  └── 'done'         → _response_to_done()  (terminal)
```

#### Response to `request`

| Case | Condition | User Response |
|---|---|---|
| 1 | Requested slot is in goal's `inform_slots` | `inform` that value |
| 2 | Requested slot is in goal's `request_slots` and already known | `inform` it from history |
| 3 | Requested slot is in goal's `request_slots` and unknown | `request` it back + piggyback a random remaining inform |
| 4 | Requested slot is irrelevant to goal | `inform 'anything'` |

#### Response to `inform`

The agent has told the user something.

* If the value **contradicts** the goal → user corrects it (`inform` the right value).
* Else → continue conversation: fulfill pending requests first, then pending informs, then say `thanks`.

#### Response to `match_found`

The agent claims it found a movie. The user checks every goal constraint against the agent's informs. If any value mismatches → `constraint_check = FAIL`, user replies `reject`. Otherwise `constraint_check = SUCCESS`, user replies `thanks`.

#### Response to `done`

If `constraint_check == SUCCESS` **and** `rest_slots` is empty → `SUCCESS`. Otherwise `FAIL`.

### Conversation Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  Episode Start                                                       │
│  User: reset() → initial 'request' action with moviename + 1 slot   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Agent responds │◄──────────────────────┐
                    └────────┬────────┘                       │
                             │                                │
            ┌────────────────▼──────────────────────┐        │
            │  User sim evaluates agent intent       │        │
            │                                        │        │
            │  request?  → inform/request back       │        │
            │  inform?   → correct or continue       │        │
            │  match_found? → accept or reject       │        │
            └────────────────┬──────────────────────-┘        │
                             │                                │
                    ┌────────▼────────┐                       │
                    │  done = False?  │──── Yes ──────────────┘
                    └────────┬────────┘
                             │ No (done or max rounds)
                    ┌────────▼────────┐
                    │  SUCCESS / FAIL │
                    └─────────────────┘
```

---

## 5. Error Model Controller (`error_model_controller.py`)

The EMC simulates noisy NLU — it corrupts the user's semantic frame *before* the StateTracker sees it, forcing the agent to be robust to misunderstandings.

Applied every turn (except the terminal one).

### Error Modes

| Mode | Effect |
|---|---|
| `0` | Replace the **value** of a random inform slot with a random value from `movie_dict` |
| `1` | Replace the entire **slot key + value** with a random slot from `movie_dict` |
| `2` | **Delete** the inform slot entirely |
| `3` | Randomly apply one of modes 0, 1, or 2 |

### Probabilities (from `constants.json`)

| Parameter | Default |
|---|---|
| `slot_error_prob` | `0.05` (5 % chance per slot) |
| `slot_error_mode` | `0` |
| `intent_error_prob` | `0.0` (intent corruption disabled by default) |

---

## 6. State Tracker & DB Query

`StateTracker` is the shared memory of the conversation. It accumulates all inform slots from both sides into `current_informs`, which is used to query the database at every turn.

### DB Query Logic (`db_query.py`)

`DBQuery` has two main operations:

**`get_db_results(constraints)`** — returns the subset of the database that satisfies *all* current constraints. Results are cached by `frozenset(constraints.items())` for speed.

**`fill_inform_slot(slot, current_informs)`** — when the agent wants to `inform` a slot, this finds the *most common* value for that slot across all DB entries still matching the current constraints. This ensures the agent always informs a consistent, constraint-satisfying value.

**`get_db_results_for_slots(current_informs)`** — returns a dict of counts used to populate the KB section of the state vector (how many DB matches remain per slot value).

---

## 7. Full Data Flow Per Turn

```
User Simulator                StateTracker              Agent            EMC
      │                            │                      │               │
      │── user_action ────────────►│                      │               │
      │                            │◄── infuse_error()────┼───────────────┤
      │                            │ (slot corruption)    │               │
      │                            │── update_state_user()│               │
      │                            │   current_informs ++ │               │
      │                            │── get_state() ──────►│               │
      │                            │   (224-dim vector)   │               │
      │                            │                      │─ DQN forward  │
      │                            │                      │  pass         │
      │                            │◄── agent_action ─────┤               │
      │                            │── update_state_agent()               │
      │                            │   fill PLACEHOLDER   │               │
      │                            │   DB query           │               │
      │◄── agent_action ───────────┤                      │               │
      │  step()                    │                      │               │
      │  → reward, done, success   │                      │               │
```

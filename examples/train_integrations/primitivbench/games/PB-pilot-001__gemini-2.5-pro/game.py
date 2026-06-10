import numpy as np
import collections

class PrimitivBenchGame:
    """
    Rival Arbiters Game.

    The player must deduce the secret goals of two NPCs, 'A' and 'B'.
    Each NPC wants to carry a specific object to a specific target location.
    The player observes their movements to infer these goals, declares support
    for one NPC, and then helps them win by pushing objects and blocking the rival.

    Primitives:
    - A1_objectness: The three objects (Gem, Key, Scroll) are unique and tied to
      specific NPC goals. They are not interchangeable.
    - A3_agency: NPCs have clear, goal-directed behavior (move to object, then
      move object to target) that the player must infer.
    - A4_geometry_topology: The game is played on a grid where pathing, adjacency
      (for pushing), and blocking are core mechanics to winning.
    """

    def __init__(self, grid_size=(5, 5), max_turns=50):
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.action_name_map = {}
        self._generate_action_map()
        self.rng = np.random.default_rng()
        self.state = None

    def _generate_action_map(self):
        self.objects = ['Gem', 'Key', 'Scroll']
        self.directions = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
        
        actions = ['WAIT', 'DECLARE_A', 'DECLARE_B']
        for d_name in self.directions:
            actions.append(f"MOVE_{d_name}")
        for d_name in self.directions:
            for obj_name in self.objects:
                actions.append(f"PUSH_{d_name}_{obj_name}")
        
        self.action_name_map = {i: name for i, name in enumerate(actions)}
        self.action_id_map = {name: i for i, name in self.action_name_map.items()}

    def reset(self, seed: int = None) -> dict:
        self.rng = np.random.default_rng(seed)
        
        # Define entities
        self.npcs = ['A', 'B']
        self.targets = ['X', 'Y']

        # Place entities
        num_entities = 1 + len(self.npcs) + len(self.objects) + len(self.targets)
        all_coords = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        chosen_coords = self.rng.choice(all_coords, size=num_entities, replace=False)
        
        coord_idx = 0
        player_pos = tuple(chosen_coords[coord_idx])
        coord_idx += 1
        
        npc_pos = {nid: tuple(chosen_coords[coord_idx + i]) for i, nid in enumerate(self.npcs)}
        coord_idx += len(self.npcs)
        
        object_pos = {oid: tuple(chosen_coords[coord_idx + i]) for i, oid in enumerate(self.objects)}
        coord_idx += len(self.objects)
        
        target_pos = {tid: tuple(chosen_coords[coord_idx + i]) for i, tid in enumerate(self.targets)}

        # Assign secret goals to NPCs
        obj_choices = self.rng.choice(self.objects, size=2, replace=False)
        target_choices = self.rng.choice(self.targets, size=2, replace=False)
        
        self._npc_goals = {
            'A': {'object': obj_choices[0], 'target': target_choices[0]},
            'B': {'object': obj_choices[1], 'target': target_choices[1]}
        }

        self.state = {
            'player_pos': player_pos,
            'npc_pos': npc_pos,
            'object_pos': object_pos,
            'target_pos': target_pos,
            'npc_carrying': {'A': None, 'B': None},
            'player_declaration': None,
            'turn': 0,
            'winner': None,
            'is_done': False,
        }
        
        return self._get_obs()

    def _get_obs(self) -> dict:
        """Returns a copy of the state, excluding secret goal info."""
        obs = {k: v for k, v in self.state.items() if k != 'winner'}
        # Deep copy mutable parts
        obs['npc_pos'] = obs['npc_pos'].copy()
        obs['object_pos'] = obs['object_pos'].copy()
        obs['target_pos'] = obs['target_pos'].copy()
        obs['npc_carrying'] = obs['npc_carrying'].copy()
        return obs

    def step(self, action) -> tuple[dict, float, bool, dict]:
        if self.state['is_done']:
            return self._get_obs(), 0.0, True, {}

        action_name = self.action_name_map.get(action, action)
        if action_name not in self.valid_actions():
            # Invalid actions result in a penalty and lost turn
            self.state['turn'] += 1
            if self.state['turn'] >= self.max_turns:
                self.state['is_done'] = True
                return self._get_obs(), -1.0, True, {'info': 'loss_max_turns'}
            return self._get_obs(), -0.1, False, {'info': f'invalid_action: {action_name}'}

        # 1. Player turn
        self._execute_player_action(action_name)
        
        # NPCs move sequentially in a round
        for npc_id in self.npcs:
            if not self.state['is_done']:
                self._npc_turn(npc_id)
                self._check_end_condition(npc_id)

        # 3. Update turn counter and check for turn limit
        self.state['turn'] += 1
        if not self.state['is_done'] and self.state['turn'] >= self.max_turns:
            self.state['is_done'] = True
            self.state['winner'] = 'loss_max_turns'
        
        reward = 0.0
        info = {}
        if self.state['is_done']:
            if self.state.get('winner') == self.state['player_declaration']:
                reward = 1.0 # Win
            else:
                reward = -1.0 # Loss
            info = {'status': self.state['winner']}
        else:
            reward = -0.01 # Small penalty per turn
            
        return self._get_obs(), reward, self.state['is_done'], info

    def _execute_player_action(self, action_name):
        if action_name == 'WAIT':
            return

        if action_name.startswith('DECLARE_'):
            self.state['player_declaration'] = action_name.split('_')[1]
            return

        parts = action_name.split('_')
        verb, d_name = parts[0], parts[1]
        
        if verb == 'MOVE':
            dx, dy = self.directions[d_name]
            px, py = self.state['player_pos']
            new_pos = (px + dx, py + dy)
            self.state['player_pos'] = new_pos
        elif verb == 'PUSH':
            obj_name = parts[2]
            dx, dy = self.directions[d_name]
            px, py = self.state['player_pos']
            obj_pos = self.state['object_pos'][obj_name]
            new_obj_pos = (obj_pos[0] + dx, obj_pos[1] + dy)
            self.state['object_pos'][obj_name] = new_obj_pos
    
    def _is_occupied(self, pos, ignore_list=None):
        if ignore_list is None:
            ignore_list = []
        
        entities = [
            ('player', self.state['player_pos']),
            ('npc_A', self.state['npc_pos']['A']),
            ('npc_B', self.state['npc_pos']['B']),
        ]
        entities.extend([(name, p) for name, p in self.state['object_pos'].items()])

        for name, entity_pos in entities:
             if name not in ignore_list and entity_pos == pos:
                 return True
        return False

    def _npc_turn(self, npc_id):
        npc_goal = self._npc_goals[npc_id]
        current_pos = self.state['npc_pos'][npc_id]
        
        # Stage 1: Move to object if not carrying it
        if self.state['npc_carrying'][npc_id] is None:
            goal_obj_name = npc_goal['object']
            target_pos = self.state['object_pos'][goal_obj_name]
            
            if current_pos == target_pos:
                # Pick up the object
                self.state['npc_carrying'][npc_id] = goal_obj_name
            else:
                # Move towards the object
                move = self._calculate_move(current_pos, target_pos, ignore_entities=[f"npc_{npc_id}"])
                self.state['npc_pos'][npc_id] = (current_pos[0] + move[0], current_pos[1] + move[1])
        
        # Stage 2: Move to target location if carrying object
        else:
            carried_obj = self.state['npc_carrying'][npc_id]
            target_loc_id = npc_goal['target']
            target_pos = self.state['target_pos'][target_loc_id]
            
            # Move towards target
            move = self._calculate_move(current_pos, target_pos, ignore_entities=[f"npc_{npc_id}", carried_obj])
            new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            self.state['npc_pos'][npc_id] = new_pos
            self.state['object_pos'][carried_obj] = new_pos # Object moves with NPC

    def _calculate_move(self, start, end, ignore_entities):
        dx, dy = end[0] - start[0], end[1] - start[1]
        
        # Try moving along the axis with greater distance first
        pri_move, sec_move = (0, 0), (0, 0)
        if abs(dx) > abs(dy):
            pri_move = (np.sign(dx), 0)
            sec_move = (0, np.sign(dy))
        else:
            pri_move = (0, np.sign(dy))
            sec_move = (np.sign(dx), 0)

        # Check if primary move is valid
        next_pos_pri = (start[0] + pri_move[0], start[1] + pri_move[1])
        if self._is_valid_move(next_pos_pri, ignore_entities):
            return pri_move
        
        # Check if secondary move is valid
        if sec_move != (0,0):
            next_pos_sec = (start[0] + sec_move[0], start[1] + sec_move[1])
            if self._is_valid_move(next_pos_sec, ignore_entities):
                return sec_move
        
        # Can't move
        return (0,0)

    def _is_valid_move(self, pos, ignore_list):
        x, y = pos
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return False
        
        # An NPC can move onto a square with its goal object/target
        # but not other entities. This logic is simplified by having NPCs
        # not move if blocked.
        occupied_by = []
        if self.state['player_pos'] == pos: occupied_by.append('player')
        for nid, npos in self.state['npc_pos'].items():
            if npos == pos: occupied_by.append(f"npc_{nid}")
        
        # An NPC can only be blocked by player or other NPC. It can move 'through' objects
        # on its way to its own goal object.
        blockers = [e for e in occupied_by if e not in ignore_list]
        return not blockers
    
    def _check_end_condition(self, last_moved_npc_id):
        npc_goal = self._npc_goals[last_moved_npc_id]
        
        is_carrying_goal_obj = self.state['npc_carrying'][last_moved_npc_id] == npc_goal['object']
        is_at_goal_target = self.state['npc_pos'][last_moved_npc_id] == self.state['target_pos'][npc_goal['target']]
        
        if is_carrying_goal_obj and is_at_goal_target:
            self.state['is_done'] = True
            declaration = self.state['player_declaration']
            
            if declaration is None:
                self.state['winner'] = f"loss_no_declaration_{last_moved_npc_id}"
            elif declaration == last_moved_npc_id:
                self.state['winner'] = last_moved_npc_id # Player Wins
            else:
                self.state['winner'] = f"loss_wrong_declaration_{last_moved_npc_id}" # Player Loses

    def valid_actions(self) -> list:
        if self.state['is_done']:
            return []

        actions = ['WAIT']
        
        # Declarations
        if self.state['player_declaration'] is None:
            actions.extend(['DECLARE_A', 'DECLARE_B'])

        # Moves
        px, py = self.state['player_pos']
        occupied_positions = set(list(self.state['npc_pos'].values()) + list(self.state['object_pos'].values()))
        
        for d_name, (dx, dy) in self.directions.items():
            new_pos = (px + dx, py + dy)
            if (0 <= new_pos[0] < self.grid_size[0] and
                0 <= new_pos[1] < self.grid_size[1] and
                new_pos not in occupied_positions):
                actions.append(f'MOVE_{d_name}')

        # Pushes
        occupied_positions.add(self.state['player_pos'])
        for o_name, o_pos in self.state['object_pos'].items():
            # cannot push an object being carried
            if o_name in self.state['npc_carrying'].values():
                continue
            
            for d_name, (dx, dy) in self.directions.items():
                if (px + dx, py + dy) == o_pos: # is adjacent
                    push_to_pos = (o_pos[0] + dx, o_pos[1] + dy)
                    if (0 <= push_to_pos[0] < self.grid_size[0] and
                        0 <= push_to_pos[1] < self.grid_size[1] and
                        push_to_pos not in occupied_positions):
                        actions.append(f'PUSH_{d_name}_{o_name}')
        
        return actions

    def is_won(self) -> bool:
        return self.state['is_done'] and self.state.get('winner') == self.state.get('player_declaration')

    def render(self, mode: str = "text") -> str:
        grid = [['.' for _ in range(self.grid_size[0])] for _ in range(self.grid_size[1])]
        
        for t_id, t_pos in self.state['target_pos'].items():
            grid[t_pos[1]][t_pos[0]] = t_id.lower()
            
        for o_id, o_pos in self.state['object_pos'].items():
            # If an NPC is at the same spot, they are carrying it. Don't double print.
            is_carried = False
            for n_id, n_pos in self.state['npc_pos'].items():
                if n_pos == o_pos and self.state['npc_carrying'][n_id] == o_id:
                    is_carried = True
                    break
            if not is_carried:
                grid[o_pos[1]][o_pos[0]] = o_id[0]

        grid[self.state['player_pos'][1]][self.state['player_pos'][0]] = 'P'
        
        for n_id, n_pos in self.state['npc_pos'].items():
            char = n_id
            if self.state['npc_carrying'][n_id] is not None:
                char = char.upper() # Capital 'A' or 'B' if carrying
            grid[n_pos[1]][n_pos[0]] = char
        
        header = f"Turn: {self.state['turn']}/{self.max_turns} | Declaration: {self.state['player_declaration']}"
        grid_str = "\n".join(" ".join(row) for row in grid)
        
        return f"{header}\n{grid_str}"

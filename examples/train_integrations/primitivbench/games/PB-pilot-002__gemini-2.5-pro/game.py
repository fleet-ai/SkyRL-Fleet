import numpy as np
import collections

class PrimitivBenchGame:
    """
    Chromatic Heist Game

    The player (P) must collect two goal items on a grid. Item value is hidden.
    The player sees items' 'raw_color' and a turn-based 'perception_key'.
    The 'true_color' of an item is 'raw_color' ^ 'perception_key'.
    The player wins by collecting the two items whose 'true_color' matches the two 'goal_true_colors'.

    Two NPCs (N1, N2) also roam the grid. They are simple-minded and are attracted
    to items based on their 'raw_color' only. They will move towards and collect
    items matching their color preference, acting as obstacles.

    This setup requires the player to:
    1. Transform perception (B1) to identify goal items.
    2. Model agency (A3) to predict and outmaneuver NPCs.
    3. Track objectness (A1) as multiple items might share a raw color but have different true colors and values.
    """
    def __init__(self):
        self.grid_size = (5, 5)
        self.num_items = 4
        self.colors = [1, 2, 3, 4]  # 1:Red, 2:Green, 3:Blue, 4:Yellow
        self._color_map = {1: 'R', 2: 'G', 3: 'B', 4: 'Y', 0: 'P'} # 0 is placeholder for collected
        self.max_steps = 50

        # These will be initialized in reset()
        self.rng = None
        self.player_pos = None
        self.npc1_pos = None
        self.npc2_pos = None
        self.npc1_preference = None
        self.npc2_preference = None
        self.items = None
        self.goal_true_colors = None
        self.perception_key = None
        self.player_inventory = None
        self.npc1_inventory = None
        self.npc2_inventory = None
        self.step_count = 0

    def reset(self, seed: int = None) -> dict:
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        # Generate unique positions for player, npcs, and items
        all_positions = set()
        while len(all_positions) < 2 + 2 + self.num_items: # Player, 2 NPCs, Items
            r = self.rng.integers(0, self.grid_size[0])
            c = self.rng.integers(0, self.grid_size[1])
            all_positions.add((r, c))
        
        pos_list = list(all_positions)
        self.rng.shuffle(pos_list)

        self.player_pos = pos_list.pop()
        self.npc1_pos = pos_list.pop()
        self.npc2_pos = pos_list.pop()

        # Initialize items
        self.items = []
        raw_colors = self.rng.choice(self.colors, self.num_items, replace=True)
        true_colors = self.rng.choice(self.colors, self.num_items, replace=False) # Ensure unique true colors

        for i in range(self.num_items):
            self.items.append({
                "id": i,
                "pos": pos_list.pop(),
                "raw_color": int(raw_colors[i]),
                "true_color": int(true_colors[i]),
                "collected_by": None # 'player', 'npc1', 'npc2'
            })

        # Define goals and preferences
        item_true_colors = [item['true_color'] for item in self.items]
        self.goal_true_colors = sorted(self.rng.choice(item_true_colors, 2, replace=False))
        
        available_raw_colors = [item['raw_color'] for item in self.items]
        prefs = self.rng.choice(available_raw_colors, 2, replace=False)
        self.npc1_preference = int(prefs[0])
        self.npc2_preference = int(prefs[1])

        self.perception_key = int(self.rng.choice(self.colors))

        # Re-calculate raw colors based on true colors and initial key to ensure solvability
        for item in self.items:
            item['raw_color'] = item['true_color'] ^ self.perception_key

        self.player_inventory = []
        self.npc1_inventory = []
        self.npc2_inventory = []
        
        return self._get_state()

    def _get_state(self) -> dict:
        # Deepcopy items to prevent external mutation
        items_state = [item.copy() for item in self.items]

        return {
            "player_pos": self.player_pos,
            "npc1_pos": self.npc1_pos,
            "npc2_pos": self.npc2_pos,
            "items": items_state,
            "perception_key": self.perception_key,
            "goal_true_colors": self.goal_true_colors,
            "npc1_preference": self.npc1_preference,
            "npc2_preference": self.npc2_preference,
            "player_inventory": [item_id for item_id in self.player_inventory],
            "step_count": self.step_count
        }

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _update_npc_state(self, npc_pos, npc_pref):
        # Find target item
        target_item = None
        min_dist = float('inf')
        
        available_items = [i for i in self.items if i['collected_by'] is None]

        # Find closest item matching preference
        for item in available_items:
            if item['raw_color'] == npc_pref:
                dist = self._manhattan_distance(npc_pos, item['pos'])
                if dist < min_dist:
                    min_dist = dist
                    target_item = item
                # Tie-breaking for determinism
                elif dist == min_dist and target_item and item['id'] < target_item['id']:
                    target_item = item
        
        new_npc_pos = npc_pos
        # Move towards target
        if target_item:
            if min_dist > 0:
                row_diff = target_item['pos'][0] - npc_pos[0]
                col_diff = target_item['pos'][1] - npc_pos[1]
                
                if abs(row_diff) > abs(col_diff):
                    new_npc_pos = (npc_pos[0] + np.sign(row_diff), npc_pos[1])
                else:
                    new_npc_pos = (npc_pos[0], npc_pos[1] + np.sign(col_diff))
            # If distance is 0, stay, collection happens after movement phase
        return new_npc_pos
        
    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if self.step_count >= self.max_steps:
            return self._get_state(), -100.0, True, {"status": "timeout"}

        self.step_count += 1
        
        # 1. Player action
        if action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            r, c = self.player_pos
            if action == "UP": r -= 1
            if action == "DOWN": r += 1
            if action == "LEFT": c -= 1
            if action == "RIGHT": c += 1
            if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                self.player_pos = (r, c)
        
        elif action == "COLLECT":
            for item in self.items:
                if item['pos'] == self.player_pos and item['collected_by'] is None:
                    item['collected_by'] = 'player'
                    self.player_inventory.append(item['id'])
                    break
        
        # 2. NPCs move
        self.npc1_pos = self._update_npc_state(self.npc1_pos, self.npc1_preference)
        self.npc2_pos = self._update_npc_state(self.npc2_pos, self.npc2_preference)

        # 3. Collection phase (NPCs)
        for item in self.items:
            if item['collected_by'] is None:
                if item['pos'] == self.npc1_pos:
                    item['collected_by'] = 'npc1'
                    self.npc1_inventory.append(item['id'])
                elif item['pos'] == self.npc2_pos:
                    item['collected_by'] = 'npc2'
                    self.npc2_inventory.append(item['id'])
        
        # 4. Check win/loss conditions
        done = False
        reward = -1.0 # Cost of a step

        player_has_goals = sorted([self.items[i]['true_color'] for i in self.player_inventory if self.items[i]['true_color'] in self.goal_true_colors])
        if player_has_goals == self.goal_true_colors:
            done = True
            reward = 100.0
            info = {"status": "win"}
            return self._get_state(), reward, done, info

        for item in self.items:
            if item['true_color'] in self.goal_true_colors and item['collected_by'] in ['npc1', 'npc2']:
                done = True
                reward = -100.0
                info = {"status": "loss"}
                return self._get_state(), reward, done, info

        if all(item['collected_by'] is not None for item in self.items):
            done = True
            reward = -100.0 # Player failed to collect goals before items ran out
            info = {"status": "loss_board_clear"}
            return self._get_state(), reward, done, info

        if self.step_count >= self.max_steps:
            done = True
            reward = -100.0
            info = {"status": "timeout"}
            return self._get_state(), reward, done, info
            
        # 5. Update perception key for next turn
        self.perception_key = int(self.rng.choice(self.colors))
        for item in self.items:
             if item['collected_by'] is None:
                item['raw_color'] = item['true_color'] ^ self.perception_key

        info = {"status": "in_progress"}
        return self._get_state(), reward, done, info

    def render(self, mode: str = "text") -> str:
        if mode != "text":
            return ""

        grid = [['.' for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        
        # Place items, their raw color is displayed
        for item in self.items:
            if item['collected_by'] is None:
                r, c = item['pos']
                grid[r][c] = self._color_map.get(item['raw_color'], '?')
        
        # Place agents
        r, c = self.npc1_pos
        grid[r][c] = '1'
        r, c = self.npc2_pos
        grid[r][c] = '2'
        r, c = self.player_pos
        grid[r][c] = 'P' if grid[r][c] == '.' else '@' # Player on top of item
        
        board_str = "\n".join("".join(row) for row in grid)

        # Info panel
        info_lines = []
        info_lines.append(f"Step: {self.step_count}")
        info_lines.append(f"Key: {self.perception_key}")
        info_lines.append(f"Goal True Colors: {self.goal_true_colors}")
        
        player_collected_true = sorted([self.items[i]['true_color'] for i in self.player_inventory])
        info_lines.append(f"Player Inventory (True Colors): {player_collected_true}")
        
        npc1_pref_char = self._color_map.get(self.npc1_preference, '?')
        npc2_pref_char = self._color_map.get(self.npc2_preference, '?')
        info_lines.append(f"NPC1 Prefers: {npc1_pref_char} | NPC2 Prefers: {npc2_pref_char}")
        
        # Item details
        info_lines.append("\nItem Details (Player Perspective):")
        for item in sorted(self.items, key=lambda x: x['id']):
            if item['collected_by'] is None:
                raw_char = self._color_map.get(item['raw_color'], '?')
                true_color_calc = item['raw_color'] ^ self.perception_key
                is_goal = "GOAL" if true_color_calc in self.goal_true_colors else ""
                info_lines.append(f"  Item {item['id']} @{item['pos']}: Raw={raw_char} -> True(calc)={true_color_calc} {is_goal}")

        return board_str + "\n" + "\n".join(info_lines)

    def valid_actions(self) -> list:
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        for item in self.items:
            if item['pos'] == self.player_pos and item['collected_by'] is None:
                actions.append("COLLECT")
                break
        return actions
    
    def is_won(self) -> bool:
        player_has_goals = sorted([self.items[i]['true_color'] for i in self.player_inventory if self.items[i]['true_color'] in self.goal_true_colors])
        return player_has_goals == self.goal_true_colors


if __name__ == '__main__':
    # Example usage for manual testing
    game = PrimitivBenchGame()
    seed = 1002 
    state = game.reset(seed=seed)
    print(game.render())

    done = False
    while not done:
        valid_acts = game.valid_actions()
        print(f"\nValid actions: {valid_acts}")
        action = input("Enter action: ").upper()
        
        if action not in valid_acts:
            print("Invalid action.")
            continue
            
        state, reward, done, info = game.step(action)
        
        print("\n" + "="*20)
        print(game.render())
        print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("GAME OVER")
            if game.is_won():
                print("YOU WON!")
            else:
                print("YOU LOST.")
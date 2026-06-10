import numpy as np

class PrimitivBenchGame:
    """
    Alchemical Ascension: A phased crafting game.

    The goal is to combine four ingredients that satisfy a cascading set of rules.
    There are 6 ingredients, each with an element, nature, and potency.
    - Phase 1 (2 ingredients): Must satisfy Rule 1 (Harmony).
    - Phase 2 (3 ingredients): Must satisfy Rule 1 (Harmony) AND Rule 2 (Stability).
    - Phase 3 (4 ingredients): Must satisfy R1, R2, AND Rule 3 (Potency) to win.
    Failing any rule check resets the cauldron.
    """

    INGREDIENTS = {
        'Ruby':     {'element': 'fire',   'nature': 'volatile', 'potency': 3},
        'Sapphire': {'element': 'water',  'nature': 'stable',   'potency': 5},
        'Emerald':  {'element': 'earth',  'nature': 'stable',   'potency': 7},
        'Topaz':    {'element': 'air',    'nature': 'volatile', 'potency': 2},
        'Obsidian': {'element': 'earth',  'nature': 'volatile', 'potency': 9},
        'Quartz':   {'element': 'water',  'nature': 'pure',     'potency': 2},
    }
    
    MAX_STEPS = 15

    def __init__(self):
        self.np_random = None
        self.cauldron = []
        self.phase = 1
        self.steps = 0
        self.won = False
        self.all_ingredient_names = sorted(list(self.INGREDIENTS.keys()))
        self.available_ingredients = []

    def _is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def _check_r1_harmony(self, ingredients_in_cauldron):
        """Rule 1: No opposing elements (fire/water or earth/air)."""
        elements = {self.INGREDIENTS[ing]['element'] for ing in ingredients_in_cauldron}
        if 'fire' in elements and 'water' in elements:
            return False
        if 'earth' in elements and 'air' in elements:
            return False
        return True

    def _check_r2_stability(self, ingredients_in_cauldron):
        """Rule 2: Volatile ingredients must not outnumber stable+pure ones."""
        natures = [self.INGREDIENTS[ing]['nature'] for ing in ingredients_in_cauldron]
        volatile_count = natures.count('volatile')
        stable_count = natures.count('stable') + natures.count('pure')
        return volatile_count <= stable_count

    def _check_r3_potency(self, ingredients_in_cauldron):
        """Rule 3: The sum of potencies must be a prime number."""
        total_potency = sum(self.INGREDIENTS[ing]['potency'] for ing in ingredients_in_cauldron)
        return self._is_prime(total_potency)

    def _reset_cauldron(self):
        self.cauldron = []
        self.available_ingredients = self.all_ingredient_names.copy()
        self.phase = 1
        return "reset"

    def reset(self, seed: int = None) -> dict:
        self.np_random = np.random.RandomState(seed)
        self.steps = 0
        self.won = False
        self._reset_cauldron()
        return self._get_state_dict()

    def _get_state_dict(self):
        return {
            "cauldron": sorted(self.cauldron),
            "available_ingredients": sorted(self.available_ingredients),
            "phase": self.phase,
            "steps_taken": self.steps,
        }

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        self.steps += 1
        done = False
        reward = -0.05  # Small penalty for taking a step
        info = {'status': ''}

        if not isinstance(action, str) or not action.startswith('add_'):
            info['status'] = f"Invalid action format: {action}. Must be 'add_<IngredientName>'."
            reward = -1.0
            done = self.steps >= self.MAX_STEPS
            return self._get_state_dict(), reward, done, info

        ingredient_name = action.split('add_')[1]

        if ingredient_name not in self.available_ingredients:
            info['status'] = f"Invalid action: {ingredient_name} is not available."
            reward = -1.0
            done = self.steps >= self.MAX_STEPS
            return self._get_state_dict(), reward, done, info
        
        # Add ingredient to cauldron
        self.cauldron.append(ingredient_name)
        self.available_ingredients.remove(ingredient_name)
        
        cauldron_size = len(self.cauldron)
        
        # Check rules based on cauldron size
        if cauldron_size == 1:
            info['status'] = 'First ingredient added.'
            reward += 0.05
        
        elif cauldron_size == 2:
            if self._check_r1_harmony(self.cauldron):
                self.phase = 2
                info['status'] = 'Phase 2 reached: Harmony rule passed.'
                reward += 0.2
            else:
                info['status'] = 'Reset: Harmony rule (R1) failed.'
                self._reset_cauldron()
                reward = -1.0

        elif cauldron_size == 3:
            if self._check_r1_harmony(self.cauldron) and self._check_r2_stability(self.cauldron):
                self.phase = 3
                info['status'] = 'Phase 3 reached: Harmony (R1) and Stability (R2) rules passed.'
                reward += 0.4
            else:
                info['status'] = 'Reset: Harmony (R1) or Stability (R2) rule failed.'
                self._reset_cauldron()
                reward = -1.0
        
        elif cauldron_size == 4:
            if (self._check_r1_harmony(self.cauldron) and
                self._check_r2_stability(self.cauldron) and
                self._check_r3_potency(self.cauldron)):
                self.won = True
                done = True
                info['status'] = 'Win! Final concoction is perfect.'
                reward = 1.0
            else:
                info['status'] = 'Reset: Final combination failed on one or more rules (R1, R2, R3).'
                self._reset_cauldron()
                reward = -1.0
        
        if self.steps >= self.MAX_STEPS:
            done = True
            if not self.won:
                info['status'] = 'Failure: Max steps reached.'

        return self._get_state_dict(), reward, done, info

    def render(self, mode: str = "text") -> str:
        if mode == "text":
            state = self._get_state_dict()
            cauldron_str = "empty"
            if state["cauldron"]:
                cauldron_str = ", ".join(state["cauldron"])
                details = []
                for ing in state['cauldron']:
                    props = self.INGREDIENTS[ing]
                    details.append(f"  - {ing}: {props['element']}, {props['nature']}, potency {props['potency']}")
                cauldron_str += "\n" + "\n".join(details)


            return (f"--- Alchemical Ascension ---\n"
                    f"Step: {self.steps}/{self.MAX_STEPS} | Phase: {self.phase}\n"
                    f"Cauldron: {cauldron_str}\n"
                    f"Available Ingredients: {', '.join(state['available_ingredients'])}\n"
                    f"Win Status: {self.won}")
        return ""

    def valid_actions(self) -> list:
        if self.won:
            return []
        return [f'add_{name}' for name in self.available_ingredients]

    def is_won(self) -> bool:
        return self.won

ok pull https://github.com/facebookresearch/end-to-end-negotiator/tree/master/src/data/negotiate  I want to set up RLVR training on this dataset. 

reward signals to ablate:
1. Terminal outcome reward (the cleanest)
At the end of a rollout, compute the agent's score: sum of item_count × item_value for items they received. Normalized to [0, 1] by dividing by 10 (the max possible). This is fully verifiable and requires no judge — just parse the final allocation and dot-product with the private value vector. No-deal = 0. If the two agents' output allocations conflict (they both claimed the same hat) = 0.

3. Pareto efficiency
You can compute whether a deal was Pareto-optimal — i.e., was there an allocation that would've made both agents better off? This is a richer verification signal that rewards quality of agreement, not just "did a deal happen." Easy to compute exhaustively since the item space is tiny.



I want to see outcome-only vs adding pareto efficiency.

Analysis: See traces in SkyRL-Fleet/skyrl-gym/skyrl_gym/envs/negotiation/eval/results, I want to make sure that the agents are actually understanding their own value functions and using them in negotiation instead of vibe-negotiating. use traces of both gpt-40-mini and qwen3.5-35b-a3b
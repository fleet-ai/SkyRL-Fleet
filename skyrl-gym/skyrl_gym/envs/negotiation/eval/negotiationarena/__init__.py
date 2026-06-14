"""Faithful reconstruction of the NegotiationArena LLM-vs-LLM negotiation harness.

NegotiationArena: How Well Can LLMs Negotiate? Platform and Analysis
(Bianchi, Chia, Yuksekgonul, Tagliabue, Jurafsky, Zou; ICML 2024; arXiv:2402.05863;
code github.com/vinid/NegotiationArena). Reimplemented from the paper + the repo's
prompt/tag scheme to evaluate trained negotiation checkpoints offline against frontier
(or scripted) LLM opponents. See SPEC.md for the protocol/equations.
"""

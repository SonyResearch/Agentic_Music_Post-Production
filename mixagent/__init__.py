"""MixAgent: reference-guided hierarchical multitrack mixing agent.

Stage 1 (:mod:`mixagent.tool_planning`) groups the stems into buses and picks every tool
chain with retrieval-augmented prompting; Stage 2 (:mod:`mixagent.param_tuning`) fills in
the parameters bottom-up with a detector -> critic -> revise loop. Rendering lives in
:mod:`mixagent.render` (processors in :mod:`mixagent.fx`).
"""

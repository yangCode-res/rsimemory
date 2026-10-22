# Actor finalization protocol v2

Version: react-final-answer-schema-v2. Applies to new executions only. Running archived-code jobs and historical scores are unchanged.

The bounded ReAct loop still uses its existing search/read/final protocol and configured maximum action rounds. On exhaustion there is still exactly one reserved actor call. That call now uses a final-answer-only system prompt plus JSON Schema requiring exactly one nonempty string field, answer (additionalProperties=false). The host labels a valid result as forced_final and records normalized_action=final. Earlier tool calls remain evidence history; no tool dispatcher is reachable in this phase.

The host independently rejects empty/whitespace/non-string answers and extra fields, even if a provider ignores the schema. Invalid structured output remains forced_final_invalid. Malformed or truncated generation remains forced_final_failed; transport service failures still propagate. No extra retry, artificial answer, altered question budget, or historical rescoring is introduced. Schema support must exist at the configured actor service; this fix does not guarantee factual correctness or prevent output truncation.

The runtime records actor_protocol_version. Baseline resume compares that field and the existing actor.py runtime hash, so old-protocol baselines cannot silently substitute for new ones. A fair new strategy comparison must use the same finalization protocol on both baselines and candidates.

Tests cover preserved ordinary tool flow, answer-only finalization, schema forwarding, invalid tool actions, extra fields, empty or whitespace/non-string answers, and truncated final output with unchanged call counts. These are protocol tests, not measured reductions of real model failure rates.

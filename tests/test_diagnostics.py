import unittest

from rsi.diagnostics import build_training_diagnostics
from rsi.evaluation import TYPES


def _rows():
    answers, queries = [], {}
    sessions = (10, 40, 80)
    for patient in range(1, 9):
        for task_index, query_type in enumerate(TYPES):
            for stage_index, session_id in enumerate(sessions):
                query_id = f"p{patient}-t{task_index}-s{session_id}"
                # Put a long initial run of one patient's records first, which
                # catches the former prefix-only sample behavior.
                score = 1 if (patient + task_index + stage_index) % 5 == 0 else 0
                answers.append({
                    "persona_id": patient, "session_id": session_id,
                    "query_id": query_id, "query_type": query_type,
                    "question": f"question {query_id}", "answer": f"answer {query_id}",
                    "score": score, "reason": "observable grader reason",
                    "evidence": [{"session_id": session_id, "text": "retrieved"}],
                    "trajectory": [{"action": "search"}, {"action": "read"}],
                    "tool_audit": [{"action": "search", "ids": [query_id]},
                                   {"action": "read", "ids": [query_id]}],
                })
                queries[query_id] = {
                    "query_type": query_type, "question": f"question {query_id}",
                    "positive_answers": ["gold"],
                    "evidence": [{"session_id": session_id, "content": "gold source"}],
                    "required_facts": ["fact"], "causal_chain": ["cause", "effect"],
                }
    # Front-load 18 questions for patient 1 to prove selection reaches beyond
    # a heavily repeated prefix when later patients add coverage.
    return answers, queries


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_coverage_selection_reaches_patients_tasks_and_time_periods_deterministically(self):
        answers, queries = _rows()
        first_run = build_training_diagnostics(answers, queries)
        second_run = build_training_diagnostics(answers, queries)

        self.assertEqual(first_run["selection"], second_run["selection"])
        self.assertEqual(first_run["summary"]["patients"], {str(i): 18 for i in range(1, 9)})
        self.assertEqual(first_run["summary"]["tasks"], {kind: 24 for kind in sorted(TYPES)})
        self.assertEqual(first_run["summary"]["time_periods"], {"early": 48, "late": 48, "middle": 48})
        self.assertEqual(first_run["selection"]["actual"], {"failures": 12, "successes": 4, "total": 16})
        self.assertGreaterEqual(set(first_run["selection"]["coverage"]["patient"]), {str(i) for i in range(1, 9)})
        self.assertEqual(set(first_run["selection"]["coverage"]["query_type"]), set(TYPES))
        self.assertEqual(set(first_run["selection"]["coverage"]["stage"]), {"early", "middle", "late"})
        self.assertTrue(all(first_run["selection"]["selection_reasons"].values()))
        self.assertEqual(first_run["summary"]["outcome_distributions"]["by_patient"]["1"]["total"], 18)
        for example in first_run["examples"]:
            self.assertIn("observation", example)
            self.assertIn("stage", example)
            self.assertIn("error_group", example)

    def test_full_examples_preserve_long_fields_and_query_annotations(self):
        answer = {
            "persona_id": 1, "session_id": 10, "query_id": "long-case",
            "query_type": "entity_exact_match", "question": "Q" * 6000,
            "answer": "A" * 7000, "score": 0, "reason": "R" * 8000,
            "evidence": [{"text": "E" * 9000}] * 11,
            "trajectory": [{"detail": "T" * 10000}] * 9,
            "tool_audit": [{"detail": "U" * 11000}] * 13,
        }
        query = {"positive_answers": ["gold" * 5000], "evidence": [{"content": "S" * 12000}],
                 "required_facts": ["F" * 13000], "causal_chain": ["C" * 14000]}
        result = build_training_diagnostics([answer], {"long-case": query})
        example = result["examples"][0]

        for field in ("question", "answer", "reason", "evidence", "trajectory", "tool_audit"):
            self.assertEqual(example[field], answer[field])
        self.assertEqual(example["reference_answers"], query["positive_answers"])
        self.assertEqual(example["source_evidence"], query["evidence"])
        self.assertEqual(example["required_facts"], query["required_facts"])
        self.assertEqual(example["causal_chain"], query["causal_chain"])

    def test_abnormal_scores_are_reported_but_not_sampled_and_missing_source_is_unknown(self):
        answers = [
            {"query_id": "fail", "persona_id": 3, "session_id": 2, "score": 0,
             "query_type": "entity_exact_match", "tool_audit": [{"action": "search"}]},
            {"query_id": "ok", "persona_id": 4, "session_id": 3, "score": 1},
            {"query_id": "abnormal-none", "persona_id": 5, "session_id": 4, "score": None},
            {"query_id": "abnormal-string", "persona_id": 6, "session_id": 5, "score": "0"},
        ]
        result = build_training_diagnostics(answers, {}, failure_limit=12, success_limit=4)

        self.assertEqual(result["summary"]["failure_count"], 1)
        self.assertEqual(result["summary"]["success_count"], 1)
        self.assertEqual(result["summary"]["abnormal_count"], 2)
        self.assertEqual(result["selection"]["actual"], {"failures": 1, "successes": 1, "total": 2})
        self.assertEqual(result["selection"]["shortfall"], {"failures": 11, "successes": 3})
        self.assertEqual(set(result["selection"]["selected_case_ids"]), {"fail", "ok"})
        failure = next(record for record in result["index"] if record["case_id"] == "fail")
        self.assertEqual(failure["observations"]["search_count"], 1)
        self.assertEqual(failure["observations"]["gold_source_session_hit_proxy"], "unknown")
        self.assertTrue(result["selection"]["uncovered"]["patient"])

    def test_forced_final_is_visible_and_source_session_overlap_is_only_a_proxy(self):
        row = {"query_id": "proxy", "persona_id": 7, "session_id": 40,
               "query_type": "state_update", "score": 0,
               "actor_status": "forced_final",
               "trajectory": [{"action": "forced_final", "status": "forced_final"}],
               "tool_audit": [{"action": "search", "ids": ["chunk"]},
                              {"action": "read", "ids": ["chunk"]}],
               "evidence": [{"session_id": 20, "text": "some retrieved text"}]}
        result = build_training_diagnostics([row], {"proxy": {"evidence": [{"session_id": 20}]}})
        record = result["index"][0]

        self.assertEqual(record["observations"]["search_count"], 1)
        self.assertEqual(record["observations"]["read_count"], 1)
        self.assertTrue(record["observations"]["forced_final"])
        self.assertIs(record["observations"]["gold_source_session_hit_proxy"], True)
        self.assertEqual(record["error_group"], "forced_final_observed")

    def test_source_session_ids_are_normalized_and_future_or_invalid_ids_are_filtered(self):
        row = {"query_id": "session-proxy", "persona_id": 1, "session_id": 40,
               "query_type": "state_update", "score": 0,
               "tool_audit": [{"action": "search", "ids": ["p1-s20-c0", "p1-s80-c1"]}],
               "evidence": [{"session_id": 20.0}, {"session_id": 80}, {"session_id": "not-a-session"}]}
        query = {"evidence": [{"session_id": "20.0"}, {"session_id": 80}, {"session_id": -1}]}
        result = build_training_diagnostics([row], {"session-proxy": query})
        observation = result["index"][0]["observations"]

        self.assertEqual(observation["gold_source_session_ids"], ["20"])
        self.assertEqual(observation["retrieved_source_session_ids"], ["20"])
        self.assertIs(observation["gold_source_session_hit_proxy"], True)

    def test_judge_reason_words_do_not_create_actor_error_group(self):
        row = {"query_id": "judge-reason", "persona_id": 1, "session_id": 10,
               "query_type": "entity_exact_match", "score": 0,
               "reason": "The answer has an error and exception in the explanation.",
               "actor_status": "completed",
               "tool_audit": [{"action": "search"}, {"action": "read"}]}
        result = build_training_diagnostics([row], {"judge-reason": {"evidence": []}})

        self.assertEqual(result["index"][0]["error_group"], "failure_other_observed")

    def test_observed_empty_search_is_a_miss_not_missing_observation(self):
        row={'query_id':'empty','persona_id':1,'session_id':10,
             'query_type':'entity_exact_match','score':0,'evidence':[],
             'tool_audit':[{'action':'search','ids':[]}]}
        query={'evidence':[{'session_id':1}]}
        report=build_training_diagnostics([row],{'empty':query})
        self.assertIs(report['index'][0]['observations']['gold_source_session_hit_proxy'],False)

    def test_unscored_actor_and_judge_anomalies_remain_distinguishable(self):
        rows=[{'query_id':'actor','score':None,'status':'actor_failed',
               'actor_status':'forced_final_invalid'},
              {'query_id':'judge','score':None,'status':'judge_failed',
               'actor_status':'completed'}]
        result=build_training_diagnostics(rows,{})
        self.assertEqual(result['summary']['abnormal_count'],2)
        self.assertEqual(result['summary']['error_groups'],
                         {'abnormal_actor_failure_observed':1,'abnormal_judge_failure_observed':1})
        self.assertEqual(result['examples'],[])


if __name__ == "__main__":
    unittest.main()

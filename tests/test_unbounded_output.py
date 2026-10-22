"""Regression tests for provider-context-bounded generations."""
import json
import unittest
from unittest.mock import patch

from rsi.llm_roles import DEFAULT_CONFIG_PATH, RoutedClient, resolve
from rsi.model import ModelClient, ModelOutputTruncated


class UnboundedOutputTests(unittest.TestCase):
    def test_default_profile_sends_expected_limits_for_all_five_roles(self):
        config = {
            'model': 'actor-model',
            'judge_model': 'judge-model',
            'actor_max_tokens': 1024,
            'optimizer_generation_settings': {
                'reflexion': {'max_tokens': 8192, 'temperature': .7, 'timeout_s': 900},
                'codegen': {'max_tokens': 16384, 'temperature': .4, 'timeout_s': 900},
            },
        }
        profile = resolve(DEFAULT_CONFIG_PATH, config)
        requests = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self, *args):
                return json.dumps({'choices': [{'message': {'content': 'ok'},
                                                'finish_reason': 'stop'}]}).encode()

        def serve(request, **kwargs):
            requests.append((request.full_url, json.loads(request.data)))
            return Response()

        with patch('urllib.request.urlopen', side_effect=serve):
            client = RoutedClient({**config, 'llm_roles': profile}, 'http://localhost/v1')
            for role in ('actor', 'judge', 'diagnosis', 'proposals', 'codegen'):
                client.complete([{'role': 'user', 'content': 'synthetic'}], role=role)

        self.assertEqual([body.get('max_tokens') for _, body in requests],
                         [4096, 1024, None, None, None])
        self.assertEqual(len(requests), 5)
        for _, body in requests[2:]:
            self.assertNotIn('max_tokens', body)

    def test_none_constructor_and_role_override_omit_limit(self):
        requests = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self, *args):
                return json.dumps({'choices': [{'message': {'content': 'ok'},
                                                'finish_reason': 'stop'}]}).encode()

        def serve(request, **kwargs):
            requests.append(json.loads(request.data))
            return Response()

        with patch('urllib.request.urlopen', side_effect=serve):
            ModelClient(model='unbounded', max_tokens=None).complete(
                [{'role': 'user', 'content': 'x'}])
            ModelClient(model='override', max_tokens=1024,
                        role_settings={'actor': {'max_tokens': None}}).complete(
                [{'role': 'user', 'content': 'x'}])
        self.assertTrue(all('max_tokens' not in body for body in requests))

    def test_invalid_client_limits_rejected_and_finite_override_retained(self):
        for value in (True, False, 0, -1, 1.5, '1024'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ModelClient(max_tokens=value)
        for value in (True, 0, -1, 1.5):
            with self.subTest(override=value):
                client = ModelClient(max_tokens=None, role_settings={'actor': {'max_tokens': value}})
                with self.assertRaises(ValueError):
                    client.complete([{'role': 'user', 'content': 'x'}])

        requests = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self, *args):
                return json.dumps({'choices': [{'message': {'content': 'ok'},
                                                'finish_reason': 'stop'}]}).encode()

        def serve(request, **kwargs):
            requests.append(json.loads(request.data))
            return Response()

        with patch('urllib.request.urlopen', side_effect=serve):
            ModelClient(max_tokens=None, role_settings={'actor': {'max_tokens': 256}}).complete(
                [{'role': 'user', 'content': 'x'}])
        self.assertEqual(requests[0]['max_tokens'], 256)

    def test_truncated_finish_reason_remains_a_distinct_error(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self, *args):
                return json.dumps({'choices': [{'message': {'content': 'partial'},
                                                'finish_reason': 'length'}]}).encode()

        with patch('urllib.request.urlopen', return_value=Response()):
            with self.assertRaises(ModelOutputTruncated):
                ModelClient(max_tokens=None).complete([{'role': 'user', 'content': 'x'}])


if __name__ == '__main__':
    unittest.main()

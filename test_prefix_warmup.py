"""Small HTTP contract tests; no accelerator or real model is used."""
import json
import tempfile
import threading
import unittest
from contextlib import contextmanager


@contextmanager
def test_directory():
    # Keep diagnostic fixtures; do not recursively delete temporary files.
    yield tempfile.mkdtemp(prefix="aisbench-prefix-test-")

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from prefix_warmup import warmup_prefixes, validate_dp_ranks, chat_completions_url, wait_for_prefix_metrics
from unittest.mock import patch


class WarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = []
        cls.fail_rank = None
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                rank = int(self.headers['X-data-parallel-rank'])
                cls.records.append((self.path, rank, body, self.headers.get('Authorization')))
                self.send_response(500 if rank == cls.fail_rank else 200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'choices': [{'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 16, 'completion_tokens': 1}}).encode())
            def log_message(self, *args):
                pass
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        type(self).records.clear()
        type(self).fail_rank = None

    def run_warmup(self, directory, dp=4, prefixes=('prefix A', 'prefix B')):
        path = Path(directory) / 'prefix.jsonl'
        path.write_text(''.join(json.dumps({'question': text})+'\n'
                               for text in prefixes for _ in range(dp)))
        report_path = Path(directory) / 'report.json'
        result = warmup_prefixes(path, dp, len(prefixes), host_ip='127.0.0.1',
            host_port=self.server.server_port, model='test', api_key='test-only-key',
            generation_kwargs={'temperature': 0, 'ignore_eos': True}, report_path=report_path)
        return result, report_path

    def test_each_prefix_reaches_each_rank_exactly_once(self):
        with test_directory() as d:
            result, report = self.run_warmup(d)
            self.assertEqual(result['completed_requests'], 8)
            self.assertEqual(len(self.records), 8)
            self.assertEqual(sorted((r[1], r[2]['messages'][0]['content']) for r in self.records),
                sorted((rank, prefix) for rank in range(4) for prefix in ('prefix A','prefix B')))
            for path, rank, body, auth in self.records:
                self.assertEqual(path, '/v1/chat/completions')
                self.assertEqual(body['max_tokens'], 1)
                self.assertIs(body['stream'], False)
                self.assertEqual(auth, 'Bearer test-only-key')
                self.assertEqual(body['messages'][0]['role'], 'user')
            self.assertEqual(json.loads(report.read_text()), result)
            self.assertNotIn('test-only-key', report.read_text())

    def test_single_rank_multiple_prefixes(self):
        with test_directory() as d:
            result, _ = self.run_warmup(d, dp=1)
            self.assertEqual(result['completed_requests'], 2)
            self.assertEqual([r[1] for r in self.records], [0, 0])

    def test_failure_aborts_and_does_not_write_success_report(self):
        type(self).fail_rank = 2
        with test_directory() as d:
            with self.assertRaisesRegex(RuntimeError, 'DP2: HTTP 500'):
                self.run_warmup(d)
            self.assertFalse((Path(d)/'report.json').exists())
            self.assertTrue(all(r[2]['messages'][0]['content']=='prefix A' for r in self.records))

    def test_dp_mismatch_is_rejected(self):
        validate_dp_ranks(4, range(4))
        for size, ranks in [(4,range(8)), (8,range(4)), (0,[])]:
            with self.assertRaises(ValueError):
                validate_dp_ranks(size, ranks)

    def test_malformed_prefix_dataset_does_not_send_requests(self):
        with test_directory() as d:
            p=Path(d)/'prefix.jsonl'
            p.write_text(json.dumps({'question':'a'})+'\n')
            with self.assertRaises(ValueError):
                warmup_prefixes(p,4,1,host_ip='127.0.0.1',host_port=self.server.server_port,model='test')
            self.assertEqual(self.records, [])

    def test_waits_for_delayed_warmup_metrics(self):
        report = {'dp_size': 2, 'requests': [
            {'rank': 0, 'prompt_tokens': 16}, {'rank': 1, 'prompt_tokens': 16}]}
        values = iter([{0: 100, 1: 200}, {0: 116, 1: 200}, {0: 116, 1: 216}])
        with patch('prefix_warmup.time.sleep') as sleep:
            current = wait_for_prefix_metrics(report, {0: 100, 1: 200}, lambda: next(values))
        self.assertEqual(current, {0: 116, 1: 216})
        self.assertEqual(sleep.call_count, 2)

    def test_metrics_timeout_stops_measured_run(self):
        report = {'dp_size': 1, 'requests': [{'rank': 0, 'prompt_tokens': 16}]}
        with self.assertRaisesRegex(RuntimeError, 'measured test was not started'):
            wait_for_prefix_metrics(report, {0: 0}, lambda: {0: 0}, timeout=0)

    def test_endpoint_urls(self):
        self.assertEqual(chat_completions_url('::1',8000), 'http://[::1]:8000/v1/chat/completions')
        self.assertEqual(chat_completions_url('ignored',1,'https://example.com/proxy/'),
                         'https://example.com/proxy/v1/chat/completions')


if __name__ == '__main__':
    unittest.main()

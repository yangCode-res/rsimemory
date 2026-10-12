import unittest
from rsi.pipeline import History, PlainRecord, make_splits, uct

class Boundaries(unittest.TestCase):
    def test_future_read_rejected(self):
        h = History(); h.append({'session_id': 10, 'content': '过去'})
        with self.assertRaises(PermissionError): h.read(20)

    def test_retrieval_filters_future(self):
        p = PlainRecord(); p.ingest({'session_id': 20, 'content': '糖尿病'})
        self.assertEqual(p.retrieve('糖尿病', 10), [])

    def test_splits_disjoint_and_complete(self):
        s = make_splits(list(range(1, 21)), 7)
        sets = list(map(set, s.values()))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(1,21)))

    def test_unvisited_priority(self):
        self.assertEqual(uct(0, 0, 1), float('inf'))

if __name__ == '__main__': unittest.main()

from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import sys
import threading
import unittest
sys.path.insert(0,str(REPOSITORY_ROOT/'scripts/dsol_paper1'))
import scripts.dsol_paper1.operations.controllers.dynamic_initial_scheduler_v1 as scheduler


class DynamicQueueTests(unittest.TestCase):
    def test_every_index_exactly_once(self):
        q=queue.Queue();expected=list(range(1000))
        for i in expected:q.put(i)
        seen=[];lock=threading.Lock()
        def record(i):
            with lock:seen.append(i)
        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(lambda _:scheduler.consume(q,record),range(32)))
        self.assertEqual(sorted(seen),expected);self.assertEqual(q.unfinished_tasks,0)

    def test_resume_does_not_enqueue_completed(self):
        done={0,2,5,9};q=queue.Queue()
        for i in range(10):
            if i not in done:q.put(i)
        seen=[];scheduler.consume(q,seen.append)
        self.assertEqual(set(seen),set(range(10))-done)

    def test_error_is_not_silently_skipped(self):
        q=queue.Queue();q.put(7)
        def fail(_):raise RuntimeError('failed episode')
        with self.assertRaises(RuntimeError):scheduler.consume(q,fail)

    def test_frozen_episode_entry_and_protocol_unchanged(self):
        self.assertTrue(scheduler.FROZEN_ENTRY.is_file())
        r=scheduler.core.release()
        self.assertEqual(r['total_episodes'],198656)
        self.assertEqual(r['workers'],32)
        self.assertEqual(str(scheduler.core.__file__),str(scheduler.FROZEN_ENTRY))
        self.assertEqual(len(r['gate_indices']),48)


if __name__=='__main__':unittest.main()

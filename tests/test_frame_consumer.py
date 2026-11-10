"""synthetic_test_only: synchronous worker lifecycle, no rendering or models."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from densekp_retrain.runtime.frame_consumer import FrameConsumer,serve_frames


class FrameConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'recipe.json').write_text('{"frame_count": 3}')
        for k in range(3):(self.root/f'frame_{k:03d}').mkdir()

    def tearDown(self):self.temp.cleanup()

    def server(self,requests,consume):
        output=io.StringIO()
        code=serve_frames(self.root,consume,incoming=io.StringIO(''.join(json.dumps(x)+'\n' for x in requests)),outgoing=output)
        return code,[json.loads(x) for x in output.getvalue().splitlines()]

    def test_failure_stops_later_frames_and_preserves_partial(self):
        def consume(root,k):
            (root/f'frame_{k:03d}'/'partial').write_text('preserve')
            raise ValueError('fixture failure')
        code,responses=self.server([{'command':'frame','frame':0},{'command':'frame','frame':1}],consume)
        self.assertEqual(code,1);self.assertEqual(responses[-1]['status'],'error')
        self.assertTrue((self.root/'frame_000/partial').exists())
        self.assertFalse((self.root/'frame_001/partial').exists())
        self.assertTrue((self.root/'consumer_failure.json').exists())

    def test_duplicate_is_rejected_without_second_call(self):
        calls=[]
        code,responses=self.server([{'command':'frame','frame':0}]*2,lambda root,k:calls.append(k))
        self.assertEqual(code,1);self.assertEqual(calls,[0]);self.assertEqual(responses[-1]['status'],'error')

    def test_eof_without_stop_is_not_success(self):
        code,_=self.server([],lambda root,k:None);self.assertEqual(code,1)

    def test_multirig_units_are_independent_and_duplicate_protected(self):
        rigs=[]
        for name in ('rig_edge','rig_side'):
            path=self.root/name;path.mkdir();(path/'frame_000').mkdir();rigs.append(dict(rig_id=name,root=str(path)))
        (self.root/'multirig.json').write_text(json.dumps(dict(rigs=rigs)))
        calls=[]
        requests=[dict(command='frame',frame=0,rig_id=name) for name in ('rig_edge','rig_side','rig_edge')]
        code,responses=self.server(requests,lambda root,k:calls.append((root.name,k)))
        self.assertEqual(calls,[('rig_edge',0),('rig_side',0)])
        self.assertEqual(code,1)
        self.assertEqual(responses[-1]['status'],'error')
        self.assertEqual(responses[-1]['completed_frames'],[[0,'rig_edge'],[0,'rig_side']])

    def test_real_pipe_reuses_one_pid_and_cleanly_stops(self):
        script='''from pathlib import Path
import sys
from densekp_retrain.runtime.frame_consumer import serve_frames
def consume(root,k):
    print("diagnostic stdout stays outside protocol")
    (root/f"frame_{k:03d}"/"done").write_text(str(k))
raise SystemExit(serve_frames(Path(sys.argv[1]),consume))
'''
        worker=FrameConsumer([sys.executable,'-c',script,str(self.root)],env=os.environ.copy(),cwd=Path.cwd(),log_path=self.root/'worker.log',deadline=time.monotonic()+10)
        try:
            responses=[worker.consume(k) for k in range(3)]
            result=worker.close()
        finally:
            if not worker.closed:worker.abort()
        self.assertEqual({r['pid'] for r in responses},{result['pid']})
        self.assertEqual(result['returncode'],0)
        self.assertTrue(all((self.root/f'frame_{k:03d}/done').exists() for k in range(3)))

    def test_timeout_and_crash_do_not_leave_child_running(self):
        for script,error in [('import time;time.sleep(5)',TimeoutError),('raise SystemExit(2)',RuntimeError)]:
            with self.assertRaises(error):
                FrameConsumer([sys.executable,'-c',script],env=os.environ.copy(),cwd=Path.cwd(),log_path=self.root/'worker.log',deadline=time.monotonic()+.2)


if __name__=='__main__':unittest.main()

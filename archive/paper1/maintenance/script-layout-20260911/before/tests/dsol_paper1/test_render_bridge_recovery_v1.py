import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/dsol_paper1'))
from recover_render_bridge_analysis_v1 import json_value


class RecoveryTest(unittest.TestCase):
    def test_numpy_scalars_preserve_values(self):
        value={'pass':np.bool_(False),'ci':[np.float64(1.25)],'count':np.int64(8)}
        self.assertEqual(json.loads(json.dumps(value,default=json_value)),{'pass':False,'ci':[1.25],'count':8})

    def test_unsupported_type_is_not_silently_stringified(self):
        with self.assertRaises(TypeError):json.dumps({'bad':object()},default=json_value)


if __name__=='__main__': unittest.main()

import unittest
from pathlib import Path
from reports.paper1.initial_results import validate_current_texts


class InitialReportTests(unittest.TestCase):
    def texts(self):
        x = ["研究结果"] * 12
        x[0] = "标准初态 32 97 198,656 无额外遮挡"
        x[1] = "旧人为遮挡与新实验不同"
        x[2] = x[3] = "历史证据"
        x[10] = "探索性 开发 初态 2、3 候选图 已查看部分测试汇总"
        return x

    def test_current_scope(self):
        validate_current_texts(self.texts())

    def test_reject_stale_headroom(self):
        for bad in ["75.88", "62.99", "主实验尚缺"]:
            t = self.texts()
            t[7] += bad
            with self.assertRaises(ValueError):
                validate_current_texts(t)

    def test_require_conditions_and_exploratory_disclosure(self):
        for i in [1, 2, 3, 10]:
            t = self.texts()
            t[i] = "结果"
            with self.assertRaises(ValueError):
                validate_current_texts(t)

    def test_current_report_loads_relocated_evidence(self):
        from reports.paper1.legacy_style import SRC, CSV
        from reports.paper1.initial_results import CurrentBrief
        archive = Path('/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908/analysis/research-v1/history/20260911T023803721133Z')
        if not CSV.exists() or not (archive / 'matrix.npz').exists():
            self.skipTest('Local historical report assets unavailable')
        self.assertTrue((SRC / 'legacy_evidence.json').is_file())
        self.assertTrue((SRC / 'new_evidence.json').is_file())
        report = CurrentBrief(archive)
        self.assertEqual(report.q.shape, (2, 32, 97))
        self.assertEqual(len(report.structure['models']), 2)
        report.save = lambda pdf: None  # Inspect plots without publishing files.
        report.space(None)
        plots = [ax for ax in report.fig.axes if ax.name == '3d']
        self.assertEqual(len(plots), 4)
        bars = [ax for ax in report.fig.axes if hasattr(ax, '_colorbar')]
        self.assertEqual(len(bars), 4)  # Exclude the report's background axes.
        from AlphaBrain.research.dsol.plotting.view_space import adaptive_limits
        for i, ax in enumerate(plots):
            values = ax.collections[0].get_array()
            expected = adaptive_limits(values, minimum_span=.001 if i >= 2 else 1.,
                                       bounds=(0., float('inf')) if i >= 2 else (0., 100.))
            self.assertEqual(ax.collections[0].get_clim(), expected)
        for left, right in [(0, 1), (2, 3)]:
            self.assertNotEqual(plots[left].collections[0].get_clim(), plots[right].collections[0].get_clim())
        import numpy as np
        np.testing.assert_allclose(plots[2].collections[0].get_array(), report.acc[0, :, 1:].mean(axis=(0, 2)))
        self.assertEqual(plots[2].collections[0].cmap.name, 'viridis_r')
        import matplotlib.pyplot as plt
        plt.close(report.fig)


if __name__ == "__main__":
    unittest.main()

import unittest
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


if __name__ == "__main__":
    unittest.main()

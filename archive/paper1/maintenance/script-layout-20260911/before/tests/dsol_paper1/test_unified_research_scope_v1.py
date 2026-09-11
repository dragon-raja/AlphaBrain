from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/dsol_paper1'))
from build_unified_research_progress_v1 import SCOPE_CONTRACT, validate_scope_texts


class ScopeTest(unittest.TestCase):
    def texts(self):
        result=['']*12
        result[0]='每个初态先对噪声求平均'
        result[1]='从任务初态到最终成败 独立复评'
        for i in range(4,11):result[i]='辅助证据 恢复快照'
        result[7]+=' 75.88% 不是标准任务初态'
        result[11]='主实验尚缺'
        return result

    def test_accepted_scope(self):
        validate_scope_texts(self.texts())
        self.assertFalse(SCOPE_CONTRACT['per_noise_hindsight_selection'])
        self.assertFalse(SCOPE_CONTRACT['dynamic_view_sequence'])
        self.assertFalse(SCOPE_CONTRACT['execution_changes_authorized_by_report_build'])

    def test_snapshot_result_cannot_be_unqualified(self):
        values=self.texts();values[7]='辅助证据 恢复快照 75.88%'
        with self.assertRaises(ValueError):validate_scope_texts(values)

    def test_every_historical_analysis_page_is_labeled(self):
        for i in range(4,11):
            values=self.texts();values[i]='一般成功率结果'
            with self.assertRaises(ValueError):validate_scope_texts(values)

    def test_old_comparison_and_completed_claim_rejected(self):
        values=self.texts();values[2]='62.99'
        with self.assertRaises(ValueError):validate_scope_texts(values)
        values=self.texts();values[11]='主实验完成'
        with self.assertRaises(ValueError):validate_scope_texts(values)


if __name__=='__main__':unittest.main()

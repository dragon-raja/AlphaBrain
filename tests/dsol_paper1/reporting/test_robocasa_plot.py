import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from AlphaBrain.research.dsol.plotting.robocasa_results import space_axes


class RoboCasaPlotTests(unittest.TestCase):
    @unittest.skipUnless(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').exists(),
                         'Report CJK font is an optional local dependency')
    def test_spatial_caption_is_additive_and_embeds_glyphs(self):
        from reports.paper1.robocasa_results import annotate_space_page, fitz
        with fitz.open() as doc:
            page=doc.new_page(width=960,height=540)
            page.insert_text((46,440),'Original conclusion stays here')
            page.insert_text((46,500),'Original source note stays here')
            annotate_space_page(page,['相对排序说明：0.168 / 0.045。','逐初态计算，不混排绝对值。'])
            text=page.get_text()
            self.assertIn('Original conclusion stays here',text)
            self.assertIn('Original source note stays here',text)
            self.assertIn('相对排序',text)
            embedded=[f for f in page.get_fonts() if f[2]=='Type3']
            self.assertTrue(embedded)
            for font in embedded:
                self.assertIn(doc.xref_get_key(font[0],'CharProcs')[0],('xref','dict'))

    def test_color_range_includes_reference_and_raw_units(self):
        fig=plt.figure()
        try:
            p=np.array([[-30,-15,.8],[30,15,1.2]])
            ax=space_axes(fig,[.1,.1,.6,.7],p,[.2,.4],'test',percentage=True,canonical=.8)
            self.assertEqual(ax.collections[0].get_clim(),(20.,80.))
            np.testing.assert_array_equal(ax.collections[0].get_array(),[20,40])
        finally:
            plt.close(fig)

    def test_accel_preserves_values_and_reverses_only_palette(self):
        fig=plt.figure()
        try:
            p=np.array([[-30,-15,.8],[30,15,1.2]])
            ax=space_axes(fig,[.1,.1,.6,.7],p,[.1,.3],'test',canonical=.2)
            np.testing.assert_allclose(ax.collections[0].get_array(),[.1,.3])
            self.assertEqual(ax.collections[0].cmap.name,'viridis_r')
        finally:
            plt.close(fig)


if __name__=='__main__':
    unittest.main()

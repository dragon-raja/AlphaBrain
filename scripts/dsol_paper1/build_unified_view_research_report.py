#!/usr/bin/env python3
"""Build one editable, source-indexed Chinese research briefing PDF.

CPU-only document generation; never imports training or evaluation launchers.
Input is a declarative JSON slide manuscript. Existing PDFs are read-only
provenance, not concatenated or overwritten. PyMuPDF is needed for final QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
INK, MUTED, BG, LINE = "#142A3D", "#526778", "#F5F7FA", "#D9E2E9"
BLUE, TEAL, ORANGE, RED = "#245D94", "#217F78", "#AC701C", "#AD4E58"
COLORS = [BLUE, TEAL, ORANGE, RED, "#6B629C", "#8A99A6"]
STATUS = {"已完成": TEAL, "历史探索": BLUE, "事后分析": ORANGE,
          "进行中": ORANGE, "待验证": RED, "研究设计": BLUE, "参考文献": MUTED}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(text, limit):
    """Deterministic mixed Chinese/Latin wrap, preserving explicit line breaks."""
    lines = []
    for paragraph in str(text).split("\n"):
        current, units = "", 0.0
        for char in paragraph:
            unit = 1.0 if ord(char) > 255 else 0.53
            if current and units + unit > limit:
                lines.append(current.rstrip())
                current, units = "", 0.0
            current += char
            units += unit
        lines.append(current.rstrip())
    return "\n".join(lines)


class Deck:
    def __init__(self, manuscript, output, qa_dir):
        self.data, self.output, self.qa_dir = manuscript, output, qa_dir
        self.text_boxes = []
        self.layout_checks = []
        self.inputs = set()
        for path in (FONT, BOLD_FONT):
            font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(FONT)).get_name()
        # System Noto CJK is an OpenType/CFF collection. Matplotlib's Type42
        # wrapper corrupts its embedded glyph mapping in common PDF viewers.
        # Type3 embeds the actual outlines while retaining Unicode extraction.
        plt.rcParams.update({"font.family": family, "pdf.fonttype": 3,
                             "axes.unicode_minus": False, "font.size": 12})

    def text(self, x, y, text, width=0.88, size=13, color=INK, bold=False, **kwargs):
        columns = width * 960 / size
        value = wrap(text, columns)
        artist = self.fig.text(x, y, value, fontsize=size, color=color, va="top",
                               fontweight="bold" if bold else "normal", linespacing=1.45, **kwargs)
        self.text_boxes.append(artist)
        return artist

    def box(self, x, y, w, h, face="white", edge=LINE):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.008",
                              facecolor=face, edgecolor=edge, linewidth=0.8, transform=self.canvas.transAxes)
        self.canvas.add_patch(patch)

    def new(self, slide, index):
        self.text_boxes = []
        self.fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
        self.canvas = self.fig.add_axes([0, 0, 1, 1])
        self.canvas.set_axis_off()
        self.canvas.add_patch(Rectangle((0, .956), 1, .044, color=INK, transform=self.canvas.transAxes))
        self.text(.045, .987, slide.get("section", "研究汇报"), .72, 9, "white")
        state = slide.get("status", "研究设计")
        self.text(.84, .987, state, .12, 9, "#DCEBF4", bold=True, ha="left")
        self.text(.047, .918, slide["title"], .905, 23, bold=True)
        self.text(.05, .85, slide.get("subtitle", ""), .90, 10.3, MUTED)
        self.text(.048, .040, "HOW TO USE CAMERA VIEWS IN VLAs  ·  " + self.data["date"], .72, 7.8, MUTED)
        self.text(.952, .040, f"{index:02d} / {len(self.data['slides']):02d}", .09, 8, MUTED, ha="right")
        sources = slide.get("sources", [])
        if sources:
            self.text(.05, .092, "依据  " + " · ".join(sources), .90, 8, MUTED)

    def lead(self, slide):
        if slide.get("lead"):
            self.box(.05, .705, .90, .09, face="#E8F0F6", edge="#C5D7E4")
            self.text(.07, .770, slide["lead"], .86, 13, BLUE, bold=True)

    def note(self, text, *, y=.19, h=.075, color=ORANGE):
        self.box(.05, y, .90, h, face="#FFF6E9", edge="#E7D9BE")
        self.text(.068, y+h-.017, text, .864, 11.2, color)

    def cards(self, slide):
        self.lead(slide)
        cards = slide["cards"]
        columns = slide.get("columns", 2)
        rows = math.ceil(len(cards)/columns)
        top = .66 if slide.get("lead") else .79
        bottom = .28 if slide.get("note") else .14
        gap = .025
        h = (top-bottom-(rows-1)*gap)/rows
        w = (.90-(columns-1)*gap)/columns
        for i, card in enumerate(cards):
            col, row = i % columns, i // columns
            x, y = .05+col*(w+gap), top-(row+1)*h-row*gap
            self.box(x, y, w, h)
            self.canvas.add_patch(Rectangle((x, y+h-.009), w, .009, color=COLORS[i % len(COLORS)], transform=self.canvas.transAxes))
            self.text(x+.018, y+h-.029, card["title"], w-.036, 14, COLORS[i % len(COLORS)], bold=True)
            artist = self.text(x+.018, y+h-.083, card["body"], w-.036, card.get("size", slide.get("body_size", 12)))
            self.layout_checks.append((artist, (x, y+.008, w, h-.07), "card-body"))
        if slide.get("note"):
            self.note(slide["note"], y=.14, h=.10)

    def table(self, slide):
        self.lead(slide)
        top = .675 if slide.get("lead") else .79
        bottom = .275 if slide.get("note") else .15
        labels, rows = slide["headers"], slide["rows"]
        widths = slide.get("widths", [1/len(labels)]*len(labels))
        size = slide.get("table_size", 11.0)
        wrapped = [[wrap(cell, width*.90*960/size-2.5) for cell,width in zip(row,widths)] for row in rows]
        headers = [wrap(v, w*.90*960/size-2.5) for v,w in zip(labels,widths)]
        weights = [max(len(c.splitlines()) for c in headers)+1.1] + [max(len(c.splitlines()) for c in r)+1.05 for r in wrapped]
        unit = (top-bottom)/sum(weights)
        y = top
        for ridx, (cells, weight) in enumerate(zip([headers]+wrapped, weights)):
            height = unit*weight
            x = .05
            for col, (cell, width) in enumerate(zip(cells, widths)):
                w = width*.90
                face = INK if ridx == 0 else ("white" if ridx % 2 else "#EDF2F6")
                self.canvas.add_patch(Rectangle((x,y-height),w,height,facecolor=face,edgecolor=BG,linewidth=1,transform=self.canvas.transAxes))
                # Cells are already wrapped above; do not wrap a second time
                # with a slightly narrower width (it creates orphan letters).
                artist = self.fig.text(x+.009, y-.014, cell, fontsize=size,
                                  va="top",linespacing=1.45,
                                  color="white" if ridx == 0 else INK,
                                  fontweight="bold" if ridx==0 else "normal")
                self.text_boxes.append(artist)
                self.layout_checks.append((artist, (x, y-height, w, height), "table-cell"))
                x += w
            y -= height
        if slide.get("note"):
            self.note(slide["note"], y=.14, h=.10)

    def bars(self, slide):
        self.lead(slide)
        ax = self.fig.add_axes([.09,.33,.55,.32 if slide.get("lead") else .42])
        categories, series = slide["categories"], slide["series"]
        x = np.arange(len(categories))
        barwidth = .78/len(series)
        for j, s in enumerate(series):
            pos = x+(j-(len(series)-1)/2)*barwidth
            bars = ax.bar(pos,s["values"],barwidth,color=s.get("color",COLORS[j]),label=s["label"])
            for b,v in zip(bars,s["values"]):
                ax.text(b.get_x()+b.get_width()/2,v+1.6,f"{v:.2f}",ha="center",fontsize=9.3,color=INK)
        ax.set_xticks(x,categories,fontsize=11)
        ax.set_ylim(*slide.get("ylim",[0,106]))
        ax.set_ylabel(slide.get("ylabel","闭环成功率（%）"),fontsize=11)
        ax.set_axisbelow(True)
        ax.grid(axis="y",color=LINE)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.01),ncol=min(3,len(series)),fontsize=9)
        self.box(.685,.32,.26,.35 if slide.get("lead") else .45)
        y=.64 if slide.get("lead") else .74
        for c in slide["callouts"]:
            self.text(.704,y,c["title"],.223,14,c.get("color",BLUE),bold=True)
            self.text(.704,y-.055,c["body"],.223,11.3)
            y-=c.get("height",.16)
        if slide.get("note"):
            self.note(slide["note"],y=.145,h=.115)

    def images(self, slide):
        self.lead(slide)
        imgs=slide["images"]
        gap=.020
        w=(.90-gap*(len(imgs)-1))/len(imgs)
        top=.66 if slide.get("lead") else .79
        for i,item in enumerate(imgs):
            path=Path(item["path"])
            if not path.is_absolute(): path=ROOT/path
            self.inputs.add(path)
            x=.05+i*(w+gap)
            self.text(x,top,item["label"],w,13,BLUE,bold=True)
            ax=self.fig.add_axes([x,top-.33,w,.275])
            ax.imshow(Image.open(path))
            ax.set_axis_off()
            if item.get("caption"):
                self.text(x,top-.365,item["caption"],w,10.7)
        if slide.get("note"):
            self.note(slide["note"],y=.145,h=.12)

    def flow(self, slide):
        self.lead(slide)
        nodes=slide["nodes"]
        gap=.035
        w=(.90-gap*(len(nodes)-1))/len(nodes)
        for i,n in enumerate(nodes):
            x=.05+i*(w+gap)
            self.box(x,.39,w,.24)
            self.text(x+.018,.60,n["title"],w-.036,14,COLORS[i],bold=True)
            self.text(x+.018,.54,n["body"],w-.036,11.3)
            if i<len(nodes)-1:
                self.canvas.add_patch(FancyArrowPatch((x+w+.002,.51),(x+w+gap-.002,.51),arrowstyle="-|>",mutation_scale=15,color=MUTED,transform=self.canvas.transAxes))
        if slide.get("equation"):
            self.text(.08,.325,slide["equation"],.84,15,BLUE,bold=True)
        if slide.get("note"):
            self.note(slide["note"],y=.145,h=.11)

    def cover(self, slide):
        self.text(.062,.765,slide["headline"],.86,30,BLUE,bold=True)
        headline_lines=len(wrap(slide["headline"],.86*960/30).splitlines())
        self.text(.065,.63 if headline_lines==1 else .58,slide["description"],.85,15)
        for i,card in enumerate(slide["stats"]):
            x=.065+i*.303
            self.box(x,.28,.273,.19)
            self.text(x+.018,.437,card["value"],.238,25,COLORS[i],bold=True)
            self.text(x+.018,.36,card["label"],.238,11.5)
        self.text(.07,.20,slide["note"],.86,11.5,MUTED)

    def save(self,pdf,index):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        issues=[]
        for artist in self.text_boxes:
            bbox=artist.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
            if bbox.x0<.015 or bbox.x1>1.01 or bbox.y0<.006 or bbox.y1>1.005:
                issues.append({"kind":"page-boundary","text":artist.get_text()[:100],"bbox":list(bbox.bounds)})
        for artist,rect,kind in self.layout_checks:
            if artist.figure is not self.fig: continue
            bbox=artist.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
            x,y,w,h=rect
            if bbox.y0<y+.002 or bbox.x1>x+w+.003:
                issues.append({"kind":kind,"text":artist.get_text()[:100],"bbox":list(bbox.bounds),"cell":rect})
        self.qa.append({"page":index,"issues":issues})
        pdf.savefig(self.fig,facecolor=BG)
        plt.close(self.fig)

    def build(self):
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.qa_dir.mkdir(parents=True,exist_ok=True)
        draft=self.qa_dir/"rendered-without-bookmarks.pdf"
        self.qa=[]
        with PdfPages(draft) as pdf:
            for i,s in enumerate(self.data["slides"],1):
                self.new(s,i)
                getattr(self,s["kind"])(s)
                self.save(pdf,i)
            pdf.infodict().update(Title=self.data["title"],Author="VLA View Research",Subject="Evidence, limitations and next experiments")
        import pymupdf
        doc=pymupdf.open(draft)
        toc=[]
        last=None
        for i,s in enumerate(self.data["slides"],1):
            if s["section"]!=last:
                toc.append([1,s["section"],i]); last=s["section"]
            toc.append([2,s["title"],i])
        doc.set_toc(toc)
        doc.set_metadata({"title":self.data["title"],"author":"VLA View Research","subject":self.data["date"]+"统一汇报；历史证据与拟贡献分开"})
        for ref in self.data.get("references", []):
            for page in doc:
                for rect in page.search_for(ref["find"]):
                    page.insert_link({"kind":pymupdf.LINK_URI,"from":rect,"uri":ref["url"]})
        # Final output is always a new report filename; historical input PDFs
        # are excluded by main() and remain byte-identical.
        doc.save(self.output,garbage=4,deflate=True)
        doc.close()
        doc=pymupdf.open(self.output)
        previews=[]
        text_pages=[]
        for i,page in enumerate(doc,1):
            path=self.qa_dir/f"page-{i:02d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(1.15,1.15)).save(path)
            previews.append(path)
            text_pages.append({"page":i,"characters":len(page.get_text()),"text":page.get_text()})
        thumbs=[]
        for start in range(0,len(previews),12):
            subset=previews[start:start+12]
            canvas=Image.new("RGB",(1280,math.ceil(len(subset)/3)*270),"#DCE2E8")
            draw=ImageDraw.Draw(canvas)
            for j,p in enumerate(subset):
                im=Image.open(p).convert("RGB"); im.thumbnail((410,235))
                x=(j%3)*425+6; y=(j//3)*270+22
                canvas.paste(im,(x,y)); draw.text((x,y-17),f"Page {start+j+1}",fill="black")
            path=self.qa_dir/f"contact-{start//12+1:02d}.jpg"; canvas.save(path,quality=90); thumbs.append(str(path))
        doc.close()
        report={"pdf":str(self.output),"pdf_sha256":sha(self.output),"page_count":len(previews),
                "layout_issues":self.qa,"text_pages":text_pages,"contact_sheets":thumbs,
                "extra_image_input_sha256":{str(p):sha(p) for p in sorted(self.inputs)}}
        (self.qa_dir/"qa.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        return report


def expanded_manuscript(path):
    data=json.loads(path.read_text())
    slides=[]
    for slide in data["slides"]:
        if slide.get("kind")=="include":
            source=(path.parent/slide["path"]).resolve()
            slides.extend(json.loads(source.read_text())["slides"])
        else:
            slides.append(slide)
    data["slides"]=slides
    return data


def export_markdown(data, output):
    lines=["# "+data["title"], "", "报告日期："+data["date"], "",
           "本文件是统一 PDF 的可编辑文字稿。事实、历史探索与待验证计划按页标注。", ""]
    for i,s in enumerate(data["slides"],1):
        lines.extend([f"## {i:02d} · {s['title']}", "", f"{s['section']}｜{s.get('status','研究设计')}", "", s.get("subtitle",""), ""])
        for field in ("headline","description","lead"):
            if s.get(field): lines.extend([s[field],""])
        if s["kind"]=="table":
            clean=lambda v:str(v).replace("\n","<br>").replace("|","\\|")
            lines.extend(["| "+" | ".join(map(clean,s["headers"]))+" |", "|"+" --- |"*len(s["headers"])])
            lines.extend("| "+" | ".join(map(clean,row))+" |" for row in s["rows"])
            lines.append("")
        if s["kind"]=="bars":
            lines.extend(["| 条件 | "+" | ".join(s["categories"])+" |", "|"+" --- |"*(len(s["categories"])+1)])
            lines.extend("| "+r["label"]+" | "+" | ".join(f"{v:.2f}" for v in r["values"])+" |" for r in s["series"])
            lines.append("")
        for group in ("cards","nodes","callouts","stats"):
            for item in s.get(group,[]):
                lines.extend(["### "+item.get("title",item.get("value","")),"",item.get("body",item.get("label","")),""])
        for item in s.get("images",[]):
            p=Path(item["path"]); p=p if p.is_absolute() else ROOT/p
            lines.extend([f"![{item['label']}]({p})", "", item.get("caption",""), ""])
        for field in ("equation","note"):
            if s.get(field): lines.extend([s[field],""])
        if s.get("sources"): lines.extend(["依据："+"；".join(s["sources"]),""])
    output.write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--qa-dir",type=Path,required=True)
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    manuscript=expanded_manuscript(args.manuscript)
    protected=[ROOT/"docs/dsol_paper1/view_revalidation_stage1_integrated_v5_20260826_zh.pdf",ROOT/"docs/dsol_paper1/view_value_new_experiments_zh.pdf"]
    if args.output.resolve() in [p.resolve() for p in protected]:
        parser.error("refusing to overwrite historical PDFs")
    before={str(p):sha(p) for p in protected}
    report=Deck(manuscript,args.output,args.qa_dir).build()
    export_markdown(manuscript,args.output.with_suffix(".md"))
    assert before=={str(p):sha(p) for p in protected},"historical PDF changed"
    receipt={"historical_inputs_sha256":before,"manuscript_sha256":sha(args.manuscript),
             "builder_sha256":sha(Path(__file__)),"pdf_sha256":report["pdf_sha256"],"pages":report["page_count"],
             "markdown_sha256":sha(args.output.with_suffix(".md")),
             "layout_issue_count":sum(len(p["issues"]) for p in report["layout_issues"]),
             "qa_directory":str(args.qa_dir.resolve()),
             "evidence_input_sha256":{str(p):sha(p) for p in
               [args.manuscript.parent/"report_sources_20260907"/name for name in
                ("legacy_evidence.json","legacy_slides.json","new_evidence.json","related_work.json","training_snapshot.json")]},
             "outline":[{"page":i,"section":s["section"],"title":s["title"],"status":s.get("status")} for i,s in enumerate(manuscript["slides"],1)]}
    (args.qa_dir/"build_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"pdf":str(args.output),"pages":receipt["pages"],"sha256":receipt["pdf_sha256"],"layout_issue_count":receipt["layout_issue_count"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

"""Build the Chinese meeting brief for the FGSCR-42 experiment ladder."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "遥感图像细粒度识别_实现算法与组会汇报说明.docx"
ASSET_DIR = ROOT / "runs" / "_analysis" / "meeting_doc_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

LETTER_W_DXA = 12240
LETTER_H_DXA = 15840
CONTENT_W_DXA = 9360
TABLE_INDENT_DXA = 120

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "243444"
MUTED = "66717D"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F4F6F9"
MID_GRAY = "D8DEE7"
PALE_GOLD = "FFF7E2"
GOLD = "9A6B00"
PALE_RED = "FDEEEE"
RED = "9B1C1C"
GREEN = "2D6A4F"

ASCII_FONT = "Calibri"
CJK_FONT = "Microsoft YaHei"
MONO_FONT = "Consolas"


RESULT_ROWS = [
    ("L0", "ResNet50, 224, basic, GAP, CE", "98.62", "90.27", "98.04", "89.88", "7.9"),
    ("L0-naive", "仅关闭分组感知切分", "98.46", "92.23", "98.29", "91.60", "7.3"),
    ("L1-320", "输入分辨率 224 -> 320", "99.31", "92.69", "98.09", "93.19", "14.0"),
    ("L1-448", "输入分辨率 224 -> 448", "99.10", "92.81", "98.27", "92.57", "26.0"),
    ("L2", "L1-448 + 遥感旋转增强", "99.79", "97.46", "99.78", "97.31", "26.0"),
    ("L3", "L2 + Compact Bilinear", "99.15", "91.62", "98.27", "91.27", "31.1"),
    ("L4", "L2 + CBAM", "99.73", "96.27", "99.78", "96.50", "27.3"),
    ("L5", "L2 + Class-Balanced CE", "99.89", "97.49", "99.82", "97.37", "23.1"),
    ("L6a", "ConvNeXt-Tiny + L2 配方", "99.89", "97.49", "99.82", "97.35", "23.4"),
    ("L6b", "ViT-B/16 + L2 配方", "99.89", "97.49", "99.82", "96.58", "50.4"),
]


REFERENCES = [
    {
        "n": 1,
        "citation": "Di, Y.; Jiang, Z.; Zhang, H. A Public Dataset for Fine-Grained Ship Classification in Optical Remote Sensing Images. Remote Sensing, 2021, 13(4):747.",
        "url": "https://doi.org/10.3390/rs13040747",
        "label": "DOI / 原论文",
    },
    {
        "n": 2,
        "citation": "He, K.; Zhang, X.; Ren, S.; Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016:770-778.",
        "url": "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
        "label": "CVF 论文页",
    },
    {
        "n": 3,
        "citation": "Gao, Y.; Beijbom, O.; Zhang, N.; Darrell, T. Compact Bilinear Pooling. CVPR, 2016:317-326.",
        "url": "https://openaccess.thecvf.com/content_cvpr_2016/html/Gao_Compact_Bilinear_Pooling_CVPR_2016_paper.html",
        "label": "CVF 论文页",
    },
    {
        "n": 4,
        "citation": "Woo, S.; Park, J.; Lee, J.-Y.; Kweon, I. S. CBAM: Convolutional Block Attention Module. ECCV, 2018:3-19.",
        "url": "https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper",
        "label": "CVF 论文页",
    },
    {
        "n": 5,
        "citation": "Cui, Y.; Jia, M.; Lin, T.-Y.; Song, Y.; Belongie, S. Class-Balanced Loss Based on Effective Number of Samples. CVPR, 2019:9268-9277.",
        "url": "https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html",
        "label": "CVF 论文页",
    },
    {
        "n": 6,
        "citation": "Liu, Z.; Mao, H.; Wu, C.-Y.; Feichtenhofer, C.; Darrell, T.; Xie, S. A ConvNet for the 2020s. CVPR, 2022:11976-11986.",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html",
        "label": "CVF 论文页",
    },
    {
        "n": 7,
        "citation": "Dosovitskiy, A. et al. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2021.",
        "url": "https://openreview.net/forum?id=YicbFdNTTy",
        "label": "OpenReview",
    },
    {
        "n": 8,
        "citation": "Loshchilov, I.; Hutter, F. Decoupled Weight Decay Regularization. ICLR, 2019.",
        "url": "https://openreview.net/forum?id=Bkg6RiCqY7",
        "label": "OpenReview",
    },
]


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def set_run_font(run, size=None, color=INK, bold=None, italic=None, mono=False):
    font = MONO_FONT if mono else ASCII_FONT
    east = MONO_FONT if mono else CJK_FONT
    run.font.name = font
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), font)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), font)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
    run._element.get_or_add_rPr().rFonts.set(qn("w:cs"), font)
    if size is not None:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = rgb(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_style_font(style, size, color=INK, bold=False):
    style.font.name = ASCII_FONT
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), ASCII_FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), ASCII_FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), CJK_FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:cs"), ASCII_FONT)
    style.font.size = Pt(size)
    style.font.color.rgb = rgb(color)
    style.font.bold = bold


def configure_styles(doc: Document):
    normal = doc.styles["Normal"]
    set_style_font(normal, 11, INK)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    h1 = doc.styles["Heading 1"]
    set_style_font(h1, 16, BLUE, True)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(10)
    h1.paragraph_format.keep_with_next = True
    h1.paragraph_format.page_break_before = False

    h2 = doc.styles["Heading 2"]
    set_style_font(h2, 13, BLUE, True)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(7)
    h2.paragraph_format.keep_with_next = True

    h3 = doc.styles["Heading 3"]
    set_style_font(h3, 12, DARK_BLUE, True)
    h3.paragraph_format.space_before = Pt(10)
    h3.paragraph_format.space_after = Pt(5)
    h3.paragraph_format.keep_with_next = True

    for style_name in ("Title", "Subtitle"):
        style = doc.styles[style_name]
        set_style_font(style, 30 if style_name == "Title" else 14,
                       DARK_BLUE if style_name == "Title" else MUTED,
                       style_name == "Title")

    caption = doc.styles["Caption"]
    set_style_font(caption, 9, MUTED)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.keep_with_next = False


def setup_page(doc: Document):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    section.different_first_page_header_footer = True
    section.start_type = WD_SECTION.NEW_PAGE
    return section


def add_page_field(paragraph):
    run = paragraph.add_run("第 ")
    set_run_font(run, 9, MUTED)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), ASCII_FONT)
    fonts.set(qn("w:hAnsi"), ASCII_FONT)
    fonts.set(qn("w:eastAsia"), CJK_FONT)
    rpr.append(fonts)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), MUTED)
    rpr.append(color)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "18")
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    paragraph._p.append(fld)
    run = paragraph.add_run(" 页")
    set_run_font(run, 9, MUTED)


def configure_headers(section):
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("遥感图像细粒度识别  |  组会技术说明")
    set_run_font(r, 8.5, MUTED, bold=True)

    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(0)
    add_page_field(p)

    first_footer = section.first_page_footer
    fp = first_footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(0)
    r = fp.add_run("组会技术说明  |  2026-08-19")
    set_run_font(r, 8.5, MUTED)


def add_numbering_definition(doc: Document, kind: str) -> int:
    root = doc.part.numbering_part.element
    abs_ids = [int(x.get(qn("w:abstractNumId"))) for x in root.findall(qn("w:abstractNum"))]
    num_ids = [int(x.get(qn("w:numId"))) for x in root.findall(qn("w:num"))]
    abstract_id = max(abs_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)

    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    lvl.append(start)
    numfmt = OxmlElement("w:numFmt")
    numfmt.set(qn("w:val"), "bullet" if kind == "bullet" else "decimal")
    lvl.append(numfmt)
    lvltext = OxmlElement("w:lvlText")
    lvltext.set(qn("w:val"), "•" if kind == "bullet" else "%1.")
    lvl.append(lvltext)
    lvl_jc = OxmlElement("w:lvlJc")
    lvl_jc.set(qn("w:val"), "left")
    lvl.append(lvl_jc)

    ppr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "540")
    tabs.append(tab)
    ppr.append(tabs)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "540")
    ind.set(qn("w:hanging"), "270")
    ppr.append(ind)
    lvl.append(ppr)

    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), ASCII_FONT)
    fonts.set(qn("w:hAnsi"), ASCII_FONT)
    fonts.set(qn("w:eastAsia"), CJK_FONT)
    rpr.append(fonts)
    lvl.append(rpr)
    abstract.append(lvl)
    # OOXML requires all abstractNum elements to precede concrete num elements.
    # Appending the abstract after existing nums makes Word ignore the custom
    # bullet format and render it as a continuing decimal list.
    first_num = root.find(qn("w:num"))
    if first_num is None:
        root.append(abstract)
    else:
        root.insert(root.index(first_num), abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abs_ref = OxmlElement("w:abstractNumId")
    abs_ref.set(qn("w:val"), str(abstract_id))
    num.append(abs_ref)
    cleanup = root.find(qn("w:numIdMacAtCleanup"))
    if cleanup is None:
        root.append(num)
    else:
        root.insert(root.index(cleanup), num)
    return num_id


def apply_numbering(paragraph, num_id: int):
    ppr = paragraph._p.get_or_add_pPr()
    numpr = ppr.find(qn("w:numPr"))
    if numpr is None:
        numpr = OxmlElement("w:numPr")
        ppr.append(numpr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    numid = OxmlElement("w:numId")
    numid.set(qn("w:val"), str(num_id))
    numpr.append(ilvl)
    numpr.append(numid)


def add_bullet(doc, text, bullet_id, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    apply_numbering(p, bullet_id)
    if bold_prefix and text.startswith(bold_prefix):
        a = p.add_run(bold_prefix)
        set_run_font(a, 11, INK, bold=True)
        b = p.add_run(text[len(bold_prefix):])
        set_run_font(b, 11, INK)
    else:
        r = p.add_run(text)
        set_run_font(r, 11, INK)
    return p


def add_numbered(doc, text, num_id, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    apply_numbering(p, num_id)
    if bold_prefix and text.startswith(bold_prefix):
        a = p.add_run(bold_prefix)
        set_run_font(a, 11, INK, bold=True)
        b = p.add_run(text[len(bold_prefix):])
        set_run_font(b, 11, INK)
    else:
        r = p.add_run(text)
        set_run_font(r, 11, INK)
    return p


def shade_paragraph(paragraph, fill, border_color=None):
    ppr = paragraph._p.get_or_add_pPr()
    shd = ppr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        ppr.append(shd)
    shd.set(qn("w:fill"), fill)
    if border_color:
        pbdr = ppr.find(qn("w:pBdr"))
        if pbdr is None:
            pbdr = OxmlElement("w:pBdr")
            ppr.append(pbdr)
        left = OxmlElement("w:left")
        left.set(qn("w:val"), "single")
        left.set(qn("w:sz"), "18")
        left.set(qn("w:space"), "7")
        left.set(qn("w:color"), border_color)
        pbdr.append(left)


def add_callout(doc, label, text, kind="info"):
    palette = {
        "info": (LIGHT_BLUE, BLUE),
        "note": (LIGHT_GRAY, DARK_BLUE),
        "warning": (PALE_GOLD, GOLD),
        "risk": (PALE_RED, RED),
    }
    fill, accent = palette[kind]
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.12)
    p.paragraph_format.right_indent = Inches(0.08)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.2
    shade_paragraph(p, fill, accent)
    r = p.add_run(label + "  ")
    set_run_font(r, 10.5, accent, bold=True)
    r = p.add_run(text)
    set_run_font(r, 10.5, INK)
    return p


def set_cell_shading(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = tcpr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcpr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, bottom=80, start=120, end=120):
    tcpr = cell._tc.get_or_add_tcPr()
    tcmar = tcpr.find(qn("w:tcMar"))
    if tcmar is None:
        tcmar = OxmlElement("w:tcMar")
        tcpr.append(tcmar)
    for side, value in (("top", top), ("bottom", bottom), ("start", start), ("end", end)):
        node = tcmar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            tcmar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color=MID_GRAY, size=5):
    tblpr = table._tbl.tblPr
    borders = tblpr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tblpr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)


def set_table_geometry(table, widths_dxa, indent=TABLE_INDENT_DXA):
    if sum(widths_dxa) != CONTENT_W_DXA:
        raise ValueError(f"table widths must sum to {CONTENT_W_DXA}: {widths_dxa}")
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tblpr = table._tbl.tblPr
    tblw = tblpr.find(qn("w:tblW"))
    if tblw is None:
        tblw = OxmlElement("w:tblW")
        tblpr.append(tblw)
    tblw.set(qn("w:type"), "dxa")
    tblw.set(qn("w:w"), str(CONTENT_W_DXA))

    tblind = tblpr.find(qn("w:tblInd"))
    if tblind is None:
        tblind = OxmlElement("w:tblInd")
        tblpr.append(tblind)
    tblind.set(qn("w:type"), "dxa")
    tblind.set(qn("w:w"), str(indent))

    layout = tblpr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tblpr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            tcpr = cell._tc.get_or_add_tcPr()
            tcw = tcpr.find(qn("w:tcW"))
            if tcw is None:
                tcw = OxmlElement("w:tcW")
                tcpr.append(tcw)
            tcw.set(qn("w:type"), "dxa")
            tcw.set(qn("w:w"), str(width))
            set_cell_margins(cell)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    trpr.append(hdr)


def prevent_row_split(row):
    trpr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:cantSplit")
    trpr.append(node)


def write_cell(cell, text, header=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=9):
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(str(text))
    set_run_font(r, size, DARK_BLUE if header else INK, bold=header)


def add_table(doc, headers, rows, widths, aligns=None, font_size=9, caption=None):
    if caption:
        p = doc.add_paragraph(style="Caption")
        p.paragraph_format.keep_with_next = True
        r = p.add_run(caption)
        set_run_font(r, 9, MUTED, bold=True)
    table = doc.add_table(rows=1, cols=len(headers))
    for j, h in enumerate(headers):
        write_cell(table.rows[0].cells[j], h, True,
                   (aligns or [WD_ALIGN_PARAGRAPH.LEFT] * len(headers))[j],
                   min(font_size, 9))
        set_cell_shading(table.rows[0].cells[j], LIGHT_BLUE)
    repeat_header(table.rows[0])
    for row_data in rows:
        row = table.add_row()
        prevent_row_split(row)
        for j, value in enumerate(row_data):
            write_cell(row.cells[j], value, False,
                       (aligns or [WD_ALIGN_PARAGRAPH.LEFT] * len(headers))[j],
                       font_size)
    set_table_geometry(table, widths)
    set_table_borders(table)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    return table


def add_hyperlink(paragraph, text, url, color=BLUE):
    rid = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), ASCII_FONT)
    fonts.set(qn("w:hAnsi"), ASCII_FONT)
    fonts.set(qn("w:eastAsia"), CJK_FONT)
    rpr.append(fonts)
    c = OxmlElement("w:color")
    c.set(qn("w:val"), color)
    rpr.append(c)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(u)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_picture(doc, path, width_inches, caption, alt_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    r = p.add_run()
    r.add_picture(str(path), width=Inches(width_inches))
    inline = doc.inline_shapes[-1]._inline
    inline.docPr.set("descr", alt_text)
    cap = doc.add_paragraph(style="Caption")
    rr = cap.add_run(caption)
    set_run_font(rr, 9, MUTED)


def add_label_value(doc, label, value, mono_value=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(label + "：")
    set_run_font(r, 10.5, DARK_BLUE, bold=True)
    r = p.add_run(value)
    set_run_font(r, 10.5, INK, mono=mono_value)
    return p


def add_equation(doc, text, note=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    shade_paragraph(p, LIGHT_GRAY)
    r = p.add_run(text)
    set_run_font(r, 11, DARK_BLUE, bold=True)
    if note:
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_after = Pt(6)
        r = q.add_run(note)
        set_run_font(r, 9, MUTED, italic=True)


def make_pipeline(path: Path):
    width, height = 2160, 585
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    bold_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    if not font_path.exists():
        font_path = Path("C:/Windows/Fonts/simhei.ttf")
    if not bold_path.exists():
        bold_path = font_path
    title_font = ImageFont.truetype(str(bold_path), 44)
    box_title_font = ImageFont.truetype(str(bold_path), 31)
    detail_font = ImageFont.truetype(str(font_path), 25)
    note_font = ImageFont.truetype(str(font_path), 25)
    labels = [
        ("1  数据审计", "尺寸/长尾/近重复", "E8EEF5", BLUE),
        ("2  可信切分", "重复组不可跨集合", "F4F6F9", DARK_BLUE),
        ("3  领域增强", "翻转 + 离散/自由旋转", "FFF7E2", GOLD),
        ("4  模型训练", "ResNet/ConvNeXt/ViT", "E8EEF5", BLUE),
        ("5  消融评价", "Top-1/MPC/F1/错误分析", "EAF5EF", GREEN),
    ]
    draw.text((width // 2, 54), "完整实现链路", font=title_font,
              fill="#" + DARK_BLUE, anchor="mm")
    box_y0, box_y1, box_w = 160, 435, 340
    xs = [38, 476, 914, 1352, 1790]
    for i, ((title, detail, fill, edge), x) in enumerate(zip(labels, xs)):
        draw.rounded_rectangle((x, box_y0, x + box_w, box_y1), radius=20,
                               fill="#" + fill, outline="#" + edge, width=4)
        draw.text((x + box_w // 2, 245), title, font=box_title_font,
                  fill="#" + edge, anchor="mm")
        draw.text((x + box_w // 2, 335), detail, font=detail_font,
                  fill="#" + INK, anchor="mm")
        if i < len(xs) - 1:
            x1, x2, y = x + box_w + 18, xs[i + 1] - 18, 297
            draw.line((x1, y, x2 - 18, y), fill="#" + MUTED, width=5)
            draw.polygon([(x2 - 20, y - 14), (x2, y), (x2 - 20, y + 14)],
                         fill="#" + MUTED)
    draw.text((width // 2, 525),
              "原则：先保证评估可信，再比较算法；每个实验相对父级只改一个关键字段",
              font=note_font, fill="#" + MUTED, anchor="mm")
    im.save(path, dpi=(180, 180))


def make_duplicate_crop(src: Path, dst: Path):
    with Image.open(src) as im:
        crop_h = min(560, im.height)
        crop = im.crop((0, 0, im.width, crop_h))
        crop.save(dst)


def add_cover(doc: Document):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(80)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("组会技术说明")
    set_run_font(r, 11, GOLD, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("遥感图像细粒度识别")
    set_run_font(r, 30, DARK_BLUE, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run("实现、算法与实验结论")
    set_run_font(r, 20, BLUE, bold=False)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(36)
    r = p.add_run("基于 FGSCR-42 的舰船细粒度分类实验")
    set_run_font(r, 13.5, MUTED)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("汇报人：__________")
    set_run_font(r, 10.5, INK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("日期：2026 年 8 月 20 日")
    set_run_font(r, 10.5, INK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("环境：PyTorch 2.11 + timm 1.0.27 + RTX 5090 D")
    set_run_font(r, 10, MUTED)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(42)
    p.paragraph_format.space_after = Pt(0)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("关键词：可信数据切分 · 旋转不变性 · 长尾学习 · 单变量消融")
    set_run_font(r, 10, DARK_BLUE, bold=True)
    doc.add_page_break()


def add_front_summary(doc, bullet_id):
    doc.add_heading("1. 一页讲清楚", level=1)
    add_callout(
        doc,
        "一句话定位",
        "本项目不是带边界框的目标检测，而是对已经裁剪出的遥感舰船图像进行 42 类细粒度分类；核心工作是先修正数据泄漏，再验证分辨率、旋转增强、二阶特征、注意力、长尾损失和主干网络的作用。",
        "info",
    )

    doc.add_heading("1.1 最值得向导师强调的四点", level=2)
    add_bullet(doc, "数据可信性：识别到公开发布数据含大量旋转/翻转近重复图；朴素分层切分时，44.78% 的验证图在训练集存在近重复孪生。", bullet_id, "数据可信性：")
    add_bullet(doc, "领域先验：遥感俯视图没有固定朝向，因此加入水平/垂直翻转、0/90/180/270 度旋转和 ±30 度自由旋转。", bullet_id, "领域先验：")
    add_bullet(doc, "长尾评价：训练集不平衡比达到 583x，不能只报 overall top-1；同步报告全类 mean-per-class、support>=5 的 MPC 和 macro-F1。", bullet_id, "长尾评价：")
    add_bullet(doc, "实验结论：旋转增强是最稳定的有效模块；Compact Bilinear 明确退化，CBAM 未带来收益；CB-CE、ConvNeXt、ViT 在 seed0 上几乎打平。", bullet_id, "实验结论：")

    pipeline = ASSET_DIR / "pipeline.png"
    make_pipeline(pipeline)
    add_picture(doc, pipeline, 6.25, "图 1  从数据审计到消融评价的完整实现链路", "遥感图像细粒度识别项目的五阶段实现流程图")

    add_callout(
        doc,
        "最稳妥的结论",
        "当前只完成 seed 0。可以说“旋转增强显著有效，L5 是当前最佳候选”，不能说“L5 已被统计证明优于其他强方案”或“达到新的 SOTA”。",
        "warning",
    )


def add_task_and_data(doc, bullet_id, num_id):
    doc.add_heading("2. 任务定义与数据处理", level=1)
    doc.add_heading("2.1 输入、输出与任务边界", level=2)
    add_label_value(doc, "输入", "一张已经裁剪出的舰船遥感图像，不包含检测框标注。")
    add_label_value(doc, "输出", "42 个舰船细分类别之一，例如驱逐舰具体舰级、航空母舰具体舰级或民用船型。")
    add_label_value(doc, "任务类型", "Fine-Grained Ship Classification / Recognition（细粒度舰船分类/识别）。")
    add_label_value(doc, "模型形式", "图像 -> 特征主干 -> 池化/注意力头 -> 42 维分类 logits -> argmax 类别。")

    add_callout(
        doc,
        "术语提醒",
        "明天汇报时请优先说“细粒度识别”或“细粒度分类”。只有在后续加入目标定位框、旋转框或实例分割后，才适合称为“细粒度检测”。",
        "risk",
    )

    doc.add_heading("2.2 数据集体检", level=2)
    data_rows = [
        ("类别数", "42", "FGSCR-42 舰船细分类别"),
        ("实得图像", "7778", "公开下载版本；论文报告约 9320 张 [1]"),
        ("重复组折叠后", "5233", "依据 dup_groups.json 的实际唯一组数"),
        ("近重复组成员", "3703（47.6%）", "1158 个多成员重复组，最大组 9 张"),
        ("朴素切分泄漏", "44.78%", "1943 张验证图中 870 张在训练集存在孪生"),
        ("原始长尾比", "389x", "最多类 778 张，最少类 2 张"),
        ("训练长尾比", "583x", "分组切分后的实际训练计数"),
        ("短边 p05/p50/p95", "64/512/1112 px", "支持 224/320/448 分辨率扫描"),
    ]
    add_table(
        doc,
        ["指标", "数值", "解释"],
        data_rows,
        [1900, 2200, 5260],
        [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        9.2,
        "表 1  FGSCR-42 数据体检摘要",
    )

    distribution = ROOT / "runs" / "_data" / "FGSCR-42" / "distribution.png"
    add_picture(doc, distribution, 6.25, "图 2  类别数量与原始图像短边分布", "FGSCR-42 类别长尾分布和图像尺寸分布直方图")

    doc.add_heading("2.3 近重复检测与分组切分", level=2)
    add_numbered(doc, "为每张图计算 64 位 aHash 与 64 位 pHash；pHash 使用 32x32 灰度图的二维 DCT 低频 8x8 区域。", num_id)
    add_numbered(doc, "对 identity、90/180/270 度旋转、水平翻转、垂直翻转六种变体分别计算哈希。", num_id)
    add_numbered(doc, "同时比较整图与中心 60% 裁剪；两种尺度的 aHash 和 pHash 汉明距离都 <=5 时才视为同一目标的近重复。", num_id)
    add_numbered(doc, "用并查集把重复图连接成原子组；分层切分时整组进入训练集或验证集，禁止同组跨集合。", num_id)
    add_numbered(doc, "每个类别至少保留 1 个验证样本，同时保证不会把一个类别全部分到验证集。", num_id)

    dup_crop = ASSET_DIR / "dup_examples_crop.png"
    make_duplicate_crop(ROOT / "runs" / "_data" / "FGSCR-42" / "dup_examples.png", dup_crop)
    add_picture(doc, dup_crop, 4.55, "图 3  检出的旋转/翻转近重复样例（节选）", "FGSCR-42 中由旋转和翻转产生的近重复舰船样例")

    add_callout(
        doc,
        "为什么这是贡献",
        "原论文 [1] 为平衡类别使用了旋转和裁剪增广；如果公开版本把这些增广图混在同一目录后再随机切分，就可能让训练/验证共享同一原图的变体。本项目用“增广感知哈希 + 组级切分”把这种风险显式控制为零。",
        "note",
    )


def add_augmentation(doc, bullet_id):
    doc.add_heading("3. 遥感领域增强与分辨率实验", level=1)
    doc.add_heading("3.1 两套增广", level=2)
    add_label_value(doc, "basic", "直接缩放到目标尺寸，训练时只做随机水平翻转；验证只做确定性 Resize + Normalize。")
    add_label_value(doc, "rs_rot", "先缩放到 1.15 倍，再做随机水平/垂直翻转、离散四向旋转、±30 度自由旋转，最后中心裁剪回目标尺寸。")
    add_bullet(doc, "1.15 倍预放大：自由旋转后中心裁剪，避免黑色三角边角进入训练样本。", bullet_id, "1.15 倍预放大：")
    add_bullet(doc, "离散旋转：随机选择 0/90/180/270 度，覆盖数据集中常见的旋转副本。", bullet_id, "离散旋转：")
    add_bullet(doc, "自由旋转：在上述方向基础上再加 ±30 度，学习连续方向扰动。", bullet_id, "自由旋转：")
    add_bullet(doc, "归一化：使用 ImageNet mean/std，与 ImageNet 预训练主干保持一致。", bullet_id, "归一化：")

    doc.add_heading("3.2 为什么扫描 224/320/448", level=2)
    p = doc.add_paragraph()
    r = p.add_run("目标像素少是遥感细粒度识别的直接困难。")
    set_run_font(r, 11, INK, bold=True)
    r = p.add_run("数据短边中位数为 512、p95 为 1112，因此 224 -> 320 -> 448 大多仍在读取真实像素，而不是纯粹上采样。实验结果显示 320 相比 224 有明显收益，但 448 的计算成本接近翻倍且 seed0 指标没有继续稳定提高。")
    set_run_font(r, 11, INK)
    add_callout(doc, "建议", "后续多种子主配方优先考虑 320；若研究重点转向极小目标局部细节，再保留 448 作为对照。", "info")


def add_model_methods(doc, bullet_id):
    doc.add_heading("4. 模型与论文方法", level=1)
    doc.add_heading("4.1 基线：预训练 ResNet50 + GAP", level=2)
    p = doc.add_paragraph()
    r = p.add_run("主干网络采用 ResNet50 [2]")
    set_run_font(r, 11, INK, bold=True)
    r = p.add_run("，通过 timm 加载 ImageNet 预训练权重，去掉原分类器与全局池化层，输出空间特征图；基线对空间维做全局平均池化（GAP），再接 42 类线性分类器。")
    set_run_font(r, 11, INK)
    add_equation(doc, "logits = W · GAP(F) + b", "F 为主干输出特征图，W/b 为新初始化的分类头参数")

    doc.add_heading("4.2 Compact Bilinear Pooling", level=2)
    p = doc.add_paragraph()
    r = p.add_run("借鉴 Gao 等人的紧凑双线性池化 [3]。")
    set_run_font(r, 11, INK, bold=True)
    r = p.add_run("完整双线性特征需要 C^2 维；对 ResNet50 的 C=2048 而言约为 419 万维。本实现使用 Tensor Sketch，把两个 Count Sketch 在频域相乘，经逆 FFT 得到 8192 维近似二阶特征，再做 signed square-root 和 L2 归一化。")
    set_run_font(r, 11, INK)
    add_equation(doc, "z = L2Norm( signed_sqrt( mean( IFFT( FFT(Ψ1(F)) × FFT(Ψ2(F)) ) ) ) )", "Ψ1、Ψ2 是由固定随机哈希与 ±1 符号映射构成的 Count Sketch")
    add_callout(doc, "实验结果", "L3 相对 L2 的全类 MPC 下降 5.84 个百分点，且每 epoch 从 26.0 s 增至 31.1 s，因此该方法已实现并完成否定性消融，但不进入最终配方。", "risk")

    doc.add_heading("4.3 CBAM 通道-空间注意力", level=2)
    p = doc.add_paragraph()
    r = p.add_run("借鉴 Woo 等人的 CBAM [4]。")
    set_run_font(r, 11, INK, bold=True)
    r = p.add_run("先对特征图做平均/最大池化生成通道描述，经共享两层 MLP 得到通道权重；再对通道维做平均/最大池化并拼接，经 7x7 卷积得到空间权重。两种注意力按顺序乘回特征图，最后再 GAP 分类。")
    set_run_font(r, 11, INK)
    add_equation(doc, "F' = Mc(F) ⊗ F；  F'' = Ms(F') ⊗ F'", "Mc 为通道注意力，Ms 为空间注意力，⊗ 为逐元素乘法")
    add_callout(doc, "实验结果", "L4 相对 L2 的 overall 下降 0.05 点、全类 MPC 下降 1.19 点；当前没有证据支持额外注意力模块。", "warning")

    doc.add_heading("4.4 主干敏感性：ConvNeXt 与 ViT", level=2)
    add_bullet(doc, "ConvNeXt-Tiny [6]：现代纯卷积主干，27.85M 参数；用于检验结论是否依赖 ResNet50。", bullet_id, "ConvNeXt-Tiny [6]：")
    add_bullet(doc, "ViT-B/16 [7]：将 448x448 图像切成 16x16 patch，共形成 28x28=784 个空间 token；代码把 token 还原为空间特征后统一接 GAP。", bullet_id, "ViT-B/16 [7]：")
    add_bullet(doc, "结果：两者与 L5 的 top-1/MPC 基本打平，但 ViT 86.28M 参数、50.4 s/epoch，计算代价最高。", bullet_id, "结果：")


def add_loss_training_metrics(doc, bullet_id):
    doc.add_heading("5. 长尾损失、训练策略与评价指标", level=1)
    doc.add_heading("5.1 Class-Balanced Cross-Entropy", level=2)
    p = doc.add_paragraph()
    r = p.add_run("借鉴 Cui 等人的 Effective Number 思想 [5]。")
    set_run_font(r, 11, INK, bold=True)
    r = p.add_run("设第 y 类训练样本数为 n_y，beta=0.9999，先计算有效样本数，再用其倒数作为类别权重并归一化到平均权重为 1。")
    set_run_font(r, 11, INK)
    add_equation(doc, "E(n_y) = (1 - β^(n_y)) / (1 - β)；  w_y ∝ 1 / E(n_y)", "β 越接近 1，权重越接近逆频率；归一化保持总体损失尺度稳定")

    p = doc.add_paragraph()
    r = p.add_run("实现中的关键修正：")
    set_run_font(r, 11, RED, bold=True)
    r = p.add_run("PyTorch 的 CrossEntropyLoss(weight=..., label_smoothing=0.1) 会把稀有类权重施加到平滑目标中的所有类别项。长尾极端时，这会让每个样本都收到巨大的稀有类惩罚。当前代码先计算逐样本平滑 CE，再仅用真实标签对应的 w_y 加权。")
    set_run_font(r, 11, INK)
    add_equation(doc, "L = Σ_i w_(y_i) · CE_smooth(z_i, y_i) / Σ_i w_(y_i)", "数值测试：smoothing=0 时与标准加权 CE 的误差为 1.19e-7")

    doc.add_heading("5.2 优化与训练配置", level=2)
    train_rows = [
        ("优化器", "AdamW [8]", "backbone lr=1e-4；新分类头 lr=1e-3"),
        ("权重衰减", "0.05", "AdamW 解耦权重衰减"),
        ("调度", "5 epoch warmup + cosine", "按 iteration 更新学习率"),
        ("训练长度", "60 epochs", "最佳 checkpoint 按全类 MPC 选择"),
        ("批量/精度", "batch 32 + bf16", "RTX 5090 D；bf16 不使用 GradScaler"),
        ("稳定性", "grad clip=1.0", "降低异常梯度影响"),
        ("复现", "deterministic=True", "Python/NumPy/Torch/CUDA RNG 均保存和恢复"),
        ("断点续训", "latest.pt 原子写入", "模型、优化器、调度器、历史与随机状态完整恢复"),
    ]
    add_table(
        doc,
        ["项目", "配置", "实现说明"],
        train_rows,
        [1900, 2200, 5260],
        [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        9.2,
        "表 2  统一训练配方",
    )

    doc.add_heading("5.3 为什么同时报告四个指标", level=2)
    add_bullet(doc, "Overall top-1：所有验证图整体准确率，容易被数量多的头部类主导。", bullet_id, "Overall top-1：")
    add_bullet(doc, "Mean-per-class top-1（MPC）：先算每类召回率再对类别平均，更关注尾部类。", bullet_id, "Mean-per-class top-1（MPC）：")
    add_bullet(doc, "MPC (support>=5)：只统计验证支持至少 5 张的类别，隔离单样本尾类造成的跳变噪声。", bullet_id, "MPC (support>=5)：")
    add_bullet(doc, "Macro-F1：每类 F1 再平均，同时惩罚“把其他类别误报成该类”的情况。", bullet_id, "Macro-F1：")
    add_callout(doc, "统计敏感性", "验证支持为 1 的类别每多判对 1 张，全类 MPC 会跳 1/42=2.38 个百分点。因此强方案必须把全类 MPC 与 support>=5 MPC 并列解读。", "warning")


def add_experiments(doc, bullet_id):
    doc.add_heading("6. 单变量消融与最终结果", level=1)
    doc.add_heading("6.1 实验阶梯", level=2)
    ladder_rows = [
        ("L0", "起点", "ResNet50 @224 + basic + GAP + CE"),
        ("L0-naive", "L0", "关闭 group-aware split，量化泄漏影响"),
        ("L1-320/448", "L0", "仅改输入分辨率，验证目标像素假设"),
        ("L2", "L1-448", "仅把 basic 改为 rs_rot"),
        ("L3", "L2", "仅把 GAP 改为 Compact Bilinear"),
        ("L4", "L2", "仅把 GAP 改为 CBAM + GAP"),
        ("L5", "L2", "仅把 CE 改为 Class-Balanced CE"),
        ("L6a/L6b", "L2", "仅把 ResNet50 改为 ConvNeXt-Tiny / ViT-B/16"),
    ]
    add_table(
        doc,
        ["实验", "父级", "唯一关键变化与目的"],
        ladder_rows,
        [900, 1500, 6960],
        [WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT],
        9.3,
        "表 3  逐级只改一个关键因素的实验设计",
    )

    doc.add_heading("6.2 seed0 完整结果", level=2)
    aligns = [
        WD_ALIGN_PARAGRAPH.CENTER,
        WD_ALIGN_PARAGRAPH.LEFT,
        WD_ALIGN_PARAGRAPH.CENTER,
        WD_ALIGN_PARAGRAPH.CENTER,
        WD_ALIGN_PARAGRAPH.CENTER,
        WD_ALIGN_PARAGRAPH.CENTER,
        WD_ALIGN_PARAGRAPH.CENTER,
    ]
    add_table(
        doc,
        ["方案", "方法", "Top-1", "全类 MPC", "MPC>=5", "Macro-F1", "秒/epoch"],
        RESULT_ROWS,
        [800, 3300, 950, 1100, 1150, 950, 1110],
        aligns,
        8.4,
        "表 4  FGSCR-42 分组感知验证结果（单位：%，除最后一列）",
    )
    add_callout(doc, "读表方式", "除 L0-naive 专门量化泄漏外，其余主结果均采用重复组不跨集合的分组感知切分。所有 checkpoint 按全类 MPC 选择。", "note")

    doc.add_heading("6.3 可以落到结论里的发现", level=2)
    add_bullet(doc, "L2 旋转增强相对 L1-448：Top-1 +0.69、全类 MPC +4.65、Macro-F1 +4.74 个百分点，是最明确的正向模块。", bullet_id, "L2 旋转增强相对 L1-448：")
    add_bullet(doc, "L1-320 相对 L0：Top-1 +0.69、全类 MPC +2.42；而 448 相比 320 只让全类 MPC +0.12，耗时却从 14.0 s 增至 26.0 s。", bullet_id, "L1-320 相对 L0：")
    add_bullet(doc, "L3 Compact Bilinear 相对 L2：全类 MPC -5.84、Macro-F1 -6.04，且更慢，明确淘汰。", bullet_id, "L3 Compact Bilinear 相对 L2：")
    add_bullet(doc, "L4 CBAM 相对 L2：全类 MPC -1.19，没有得到正向证据。", bullet_id, "L4 CBAM 相对 L2：")
    add_bullet(doc, "L5、ConvNeXt、ViT 的 Top-1/全类 MPC/MPC>=5 几乎相同；L5 参数更少、速度更快、Macro-F1 最高，因此是当前主候选。", bullet_id, "L5、ConvNeXt、ViT：")

    curve = ROOT / "runs" / "L5_cb_ce" / "seed0" / "curves.png"
    add_picture(doc, curve, 6.2, "图 4  L5 的损失与验证指标曲线", "L5 类别平衡交叉熵实验的训练损失和验证准确率曲线")

    errors = ROOT / "runs" / "_analysis" / "errors_L5_cb_ce.png"
    add_picture(doc, errors, 6.15, "图 5  L5 验证集仅有的两张误分类样本", "L5 最佳模型的两张误分类舰船图像及真实和预测标签")


def add_paper_mapping(doc, bullet_id):
    doc.add_heading("7. 论文对应关系：哪些方法来自哪里", level=1)
    add_callout(doc, "正确表述", "建议说“本项目实现并验证了论文中的方法”，不要说“这些算法是我提出的”。自主部分主要是数据审计、增广感知近重复检测、分组切分、实验阶梯和针对 label smoothing 的损失实现修正。", "info")

    mappings = [
        ("[1] FGSCR-42 数据集论文", "采用其 42 类遥感舰船细粒度任务；论文也使用旋转/裁剪增广。项目进一步发现公开下载版本数量不一致和近重复泄漏，并重新设计切分。", "可以讲：数据集与遥感增强动机来自该论文，数据质量审计与分组防泄漏是我的实现。"),
        ("[2] ResNet", "作为 L0-L5 的预训练特征主干，用残差连接支持深层特征学习。", "可以讲：以标准 ResNet50 作为可信基线，不手写主干，通过 timm 加载预训练权重并全量微调。"),
        ("[3] Compact Bilinear Pooling", "按论文的 Tensor Sketch 思路实现 Count Sketch + FFT 的 8192 维二阶池化。", "可以讲：我完整实现并做了消融，但在本数据上性能下降，所以最终没有保留。"),
        ("[4] CBAM", "实现顺序通道注意力和空间注意力，置于分类池化之前。", "可以讲：用于弱监督强调判别区域；本数据 seed0 没有收益，说明额外注意力不是必然有效。"),
        ("[5] Class-Balanced Loss", "使用有效样本数公式进行类别重加权，beta=0.9999。", "可以讲：这是最终候选 L5 的核心论文方法；我额外处理了类权重与 label smoothing 的实现冲突。"),
        ("[6] ConvNeXt", "替换 ResNet50 做现代卷积主干敏感性实验。", "可以讲：验证方法结论不是 ResNet 特例，但 seed0 没有证明值得增加参数。"),
        ("[7] Vision Transformer", "使用 ViT-B/16 作为 Transformer 主干对照。", "可以讲：高分辨率下 784 tokens，性能打平但计算成本明显更高。"),
        ("[8] AdamW", "训练优化器，权重衰减与损失梯度更新解耦。", "可以讲：采用 AdamW + warmup + cosine 的统一微调配方。"),
    ]
    for title, use, speech in mappings:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        set_run_font(r, 11, DARK_BLUE, bold=True)
        add_bullet(doc, "代码中怎么用：" + use, bullet_id, "代码中怎么用：")
        add_bullet(doc, "汇报时怎么说：" + speech, bullet_id, "汇报时怎么说：")


def add_script_and_qa(doc, bullet_id, num_id):
    doc.add_heading("8. 明天可直接照着讲的 5 分钟稿", level=1)
    script = [
        ("第 1 分钟：任务", "我的任务是 FGSCR-42 上的遥感舰船 42 类细粒度识别。输入是裁好的单舰图像，输出是具体舰级或船型，所以它本质上是细粒度分类，而不是带框检测。困难主要有目标像素少、方向任意、背景干扰和类别长尾。"),
        ("第 2 分钟：数据可信性", "我先没有直接堆模型，而是做数据审计。公开下载版有 7778 张图，其中 3703 张属于近重复组。若按普通分层随机切分，44.78% 的验证图在训练集有旋转或翻转孪生。因此我用 aHash+pHash、六种方向变体和双尺度中心裁剪检出重复组，再保证同一组只进入一个集合。"),
        ("第 3 分钟：算法", "基线是 ImageNet 预训练 ResNet50。针对遥感方向任意，我加入水平/垂直翻转、四向旋转和 ±30 度自由旋转；针对背景干扰，我分别实现了论文中的 Compact Bilinear Pooling 和 CBAM；针对 583 倍长尾，我实现了基于有效样本数的 Class-Balanced Cross-Entropy；还用 ConvNeXt-Tiny 和 ViT-B/16 做主干敏感性对照。"),
        ("第 4 分钟：实验设计", "我采用实验阶梯，每一级相对父级只改一个关键字段。评价时不只看 overall top-1，还看全类 mean-per-class、支持至少 5 张类别的 MPC 和 macro-F1；checkpoint 按全类 MPC 选择。这样能避免头部类高准确率掩盖尾类失败。"),
        ("第 5 分钟：结论", "seed0 结果表明，旋转增强是最清晰的有效模块，相对 448 基线全类 MPC 提升 4.65 点。Compact Bilinear 和 CBAM 都没有带来收益。L5 的 Class-Balanced CE 达到 99.89 top-1、97.49 全类 MPC 和 97.37 macro-F1，是当前最佳候选，但 L5、ConvNeXt、ViT 差异很小，所以我下一步会补 seeds 1 和 2，报告 mean±std 后再下统计结论。"),
    ]
    for title, text in script:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        set_run_font(r, 11.5, BLUE, bold=True)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.15)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.25
        r = p.add_run(text)
        set_run_font(r, 10.5, INK)

    doc.add_heading("9. 导师可能追问与回答", level=1)
    qa = [
        ("Q1：为什么不直接沿用数据集官方切分？", "公开下载版实际文件与论文报告的 9320 张不一致，而且当前目录结构没有可靠的原始切分信息。更关键的是存在旋转/翻转近重复，因此我先构建重复组，再做可复现的组级分层切分。"),
        ("Q2：44.78% 泄漏为什么没有让 overall 一定变高？", "泄漏会改变样本依赖关系，但指标变化还受每类验证支持、切分难度和尾类组成影响。对照中 naive split 的 overall 反而低 0.16 点，而全类 MPC 高 1.96 点，所以我只说它让评估不再独立，不把所有高分都归因于泄漏。"),
        ("Q3：为什么旋转增强有效？", "俯视遥感图没有固定上方向，同一舰船在不同航向下类别不变。rs_rot 直接把这种标签不变性编码进训练，且全类 MPC 和 macro-F1 都提升约 4.7 点。"),
        ("Q4：为什么 448 不如 320？", "更高分辨率增加细节，也增加计算、过拟合和插值噪声。seed0 中 448 对 320 没有性价比优势；需要多种子确认后再决定主分辨率。"),
        ("Q5：为什么 Compact Bilinear / CBAM 没有效果？", "论文方法有适用条件。当前基线加旋转增强后已经接近饱和，且数据量小、背景与目标关系复杂；额外模块可能增加优化难度或放大背景共现。消融的价值正是用实验排除不合适的模块。"),
        ("Q6：L5 为什么只提升一点？", "强方案在 support>=5 的 30 类上已经约 99.8%，存在明显天花板；全类 MPC 又受单样本尾类强烈影响。当前提升是候选信号，必须用 seeds 1/2 的均值和标准差判断。"),
        ("Q7：能否与论文中的 9320 张结果直接比较？", "不能严格直接比较。我的公开版本只有 7778 张，并且我采用自定义组级切分；数据规模和协议不同。可以对照量级，但不能据此宣称超越论文。"),
        ("Q8：你的工作里哪些是自主实现？", "数据质量审计、增广感知双哈希去重、组级分层切分、单变量实验阶梯、support>=5 指标，以及 Class-Balanced CE 与 label smoothing 交互问题的修正。主干、CBAM、紧凑双线性和有效样本数思想分别来自对应论文。"),
    ]
    for q, a in qa:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(q)
        set_run_font(r, 10.8, DARK_BLUE, bold=True)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.2
        r = p.add_run("A：" + a)
        set_run_font(r, 10.2, INK)

    doc.add_heading("10. 表述边界与下一步", level=1)
    add_callout(doc, "不要这样说", "“我做了细粒度目标检测”“CBAM 提升了性能”“L5 已经统计显著优于 ViT”“我的结果直接超过论文”。这些说法分别与任务标注、实验结果、种子数量和数据协议不符。", "risk")
    add_bullet(doc, "补跑 seeds 1、2，输出 mean ± std，并重点比较 L2、L5、L6a；L3/L4 可不再扩展。", bullet_id)
    add_bullet(doc, "若优先考虑效率，补做 L1-320 + rs_rot + CB-CE，验证 320 是否能接近 448 强方案。", bullet_id)
    add_bullet(doc, "针对两张稳定误分样本，做 Grad-CAM/注意力可视化，判断模型看的是舰体结构还是背景。", bullet_id)
    add_bullet(doc, "若导师要求真正的“检测”，下一阶段需换用带旋转框标注的数据集，并接入 oriented detector；当前分类代码不能直接宣称完成检测。", bullet_id)


def add_code_map_and_refs(doc, bullet_id):
    doc.add_heading("11. 代码位置速查", level=1)
    code_rows = [
        ("数据与切分", "src/data.py", "ImageFolder、重复组映射、分组分层切分、可选平衡采样"),
        ("数据增强", "src/transforms.py", "basic 与 rs_rot；1.15x 预放大和中心裁剪"),
        ("模型", "src/models.py", "timm 主干、GAP、Compact Bilinear、CBAM、ViT token 还原"),
        ("长尾损失", "src/losses.py", "Effective Number 权重与 SampleWeightedSmoothedCE"),
        ("训练循环", "src/engine.py", "bf16、warmup+cosine、梯度裁剪、验证"),
        ("指标", "src/metrics.py", "混淆矩阵、Top-1、MPC、Macro-F1、MPC>=5"),
        ("单实验入口", "tools/train.py", "AdamW、best/latest checkpoint、随机状态精确恢复"),
        ("批量阶梯", "tools/run_ladder.py", "串行排队、跳过已完成、从 latest.pt 恢复"),
        ("汇总分析", "tools/analyze.py", "对比表、父级差异、混淆矩阵和误差图"),
        ("去重审计", "tools/dedup_check.py", "aHash+pHash、六方向、双尺度、并查集、泄漏率"),
    ]
    add_table(
        doc,
        ["模块", "文件", "主要功能"],
        code_rows,
        [1700, 2200, 5460],
        [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
        9.0,
        "表 5  实现与代码对应关系",
    )

    doc.add_page_break()
    doc.add_heading("12. 参考论文", level=1)
    for ref in REFERENCES:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.28)
        p.paragraph_format.first_line_indent = Inches(-0.28)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(f"[{ref['n']}] {ref['citation']}  ")
        set_run_font(r, 9.8, INK)
        add_hyperlink(p, ref["label"], ref["url"])

    add_callout(doc, "论文使用原则", "报告中只把代码真实实现的方法对应到论文。旋转增强与 FGSCR-42 论文的领域动机一致，但具体的六方向双哈希去重、1.15x 旋转裁剪与分组切分属于本项目工程设计。", "note")


def add_document_properties(doc: Document):
    props = doc.core_properties
    props.title = "遥感图像细粒度识别：实现、算法与实验结论"
    props.subject = "FGSCR-42 舰船细粒度分类组会技术说明"
    props.author = ""
    props.keywords = "FGSCR-42, fine-grained ship classification, ResNet, CBAM, class-balanced loss"
    props.comments = "Generated from the verified experiment artifacts in D:\\rs-project\\fgvc."


def validate_sources():
    required = [
        ROOT / "REPORT.md",
        ROOT / "runs" / "_analysis" / "TABLE.md",
        ROOT / "runs" / "_data" / "FGSCR-42" / "distribution.png",
        ROOT / "runs" / "_data" / "FGSCR-42" / "dup_examples.png",
        ROOT / "runs" / "L5_cb_ce" / "seed0" / "curves.png",
        ROOT / "runs" / "_analysis" / "errors_L5_cb_ce.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))

    metrics = json.loads((ROOT / "runs" / "L5_cb_ce" / "seed0" / "metrics.json").read_text(encoding="utf-8"))
    expected = (99.89, 97.49, 97.37)
    actual = (
        round(metrics["overall_top1"] * 100, 2),
        round(metrics["mean_per_class_top1"] * 100, 2),
        round(metrics["macro_f1"] * 100, 2),
    )
    if actual != expected:
        raise ValueError(f"L5 metrics changed; expected {expected}, got {actual}")


def build():
    validate_sources()
    doc = Document()
    configure_styles(doc)
    section = setup_page(doc)
    configure_headers(section)
    add_document_properties(doc)

    bullet_id = add_numbering_definition(doc, "bullet")
    num_id = add_numbering_definition(doc, "decimal")

    add_cover(doc)
    add_front_summary(doc, bullet_id)
    add_task_and_data(doc, bullet_id, num_id)
    add_augmentation(doc, bullet_id)
    add_model_methods(doc, bullet_id)
    add_loss_training_metrics(doc, bullet_id)
    add_experiments(doc, bullet_id)
    add_paper_mapping(doc, bullet_id)
    add_script_and_qa(doc, bullet_id, num_id)
    add_code_map_and_refs(doc, bullet_id)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()

import os
import sys
import re
import html
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

MATH_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

LATEX_SYMBOLS = {
    r'\Delta': 'Δ',
    r'\alpha': 'α',
    r'\beta': 'β',
    r'\lambda': 'λ',
    r'\mu': 'μ',
    r'\sigma': 'σ',
    r'\tau': 'τ',
    r'\omega': 'ω',
    r'\pi': 'π',
    r'\epsilon': 'ε',
    r'\pm': '±',
    r'\mp': '∓',
    r'\times': '×',
    r'\cdot': '⋅',
    r'\odot': '⊙',
    r'\approx': '≈',
    r'\sim': '~',
    r'\le': '≤',
    r'\ge': '≥',
    r'\neq': '≠',
    r'\in': '∈',
    r'\infty': '∞',
    r'\partial': '∂',
    r'\rightarrow': '→',
    r'\to': '→',
    r'\dots': '…',
    r'\ldots': '…',
    r'\flat': '♭',
    r'\sharp': '♯',
    r'\mid': ' | ',
    r'\|': '‖',
    r'\{': '⦃',
    r'\}': '⦄',
    r'\%': '%',
    r'\quad': '    ',
    r'\,': ' ',
    r'\mathcal{L}': 'L',
    r'\mathcal{G}': 'G',
    r'\mathcal{Q}': 'Q',
    r'\mathbb{R}': 'ℝ',
    r'\mathbb{I}': '𝕀',
    r'\left(': '(',
    r'\right)': ')',
    r'\left[': '[',
    r'\right]': ']',
    r'\left\{': '{',
    r'\right\}': '}',
    r'\tanh': 'tanh',
    r'\exp': 'exp',
    r'\log': 'log',
    r'\cos': 'cos',
    r'\sin': 'sin',
    r'\max': 'max',
    r'\min': 'min',
    r'\text{softmax}': 'softmax',
    r'\text{LayerNorm}': 'LayerNorm',
    r'\text{Attention}': 'Attention',
    r'\text{Swish}': 'Swish',
    r'\text{Output}': 'Output',
    r'\text{FFN}': 'FFN',
    r'\text{MHSA}': 'MHSA',
    r'\text{Conv}': 'Conv',
    r'\text{GRL}': 'GRL',
    r'\text{Quantile}': 'Quantile',
    r'\text{MRR}': 'MRR',
    r'\text{Recall}': 'Recall',
    r'\text{rank}': 'rank',
    r'\text{const}': 'const',
}

def set_style_font_strict(style, font_name="Times New Roman", size_pt=12, color_rgb=(0, 0, 0)):
    style.font.name = font_name
    style.font.size = Pt(size_pt)
    style.font.color.rgb = RGBColor(*color_rgb)
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    for attr in ['w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme']:
        if qn(attr) in rFonts.attrib:
            del rFonts.attrib[qn(attr)]
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)
    rFonts.set(qn('w:eastAsia'), font_name)
    rFonts.set(qn('w:cs'), font_name)

def set_run_font_strict(run, font_name="Times New Roman", size_pt=12, bold=False, italic=False, subscript=False, superscript=False, color_rgb=(0, 0, 0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic
    run.font.subscript = subscript
    run.font.superscript = superscript
    if color_rgb is not None:
        run.font.color.rgb = RGBColor(*color_rgb)
    else:
        run.font.color.rgb = RGBColor(0, 0, 0)
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    for attr in ['w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme']:
        if qn(attr) in rFonts.attrib:
            del rFonts.attrib[qn(attr)]
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)
    rFonts.set(qn('w:eastAsia'), font_name)
    rFonts.set(qn('w:cs'), font_name)

def extract_braced(s, idx):
    """Given s[idx] == '{', return (content_inside_braces, next_idx)."""
    assert s[idx] == '{'
    depth = 0
    start = idx + 1
    i = idx
    while i < len(s):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return s[start:], len(s)

def extract_atom(s, idx):
    """Extract a single math token/group starting at idx for _ or ^ or frac."""
    while idx < len(s) and s[idx] == ' ':
        idx += 1
    if idx >= len(s):
        return "", idx
    if s[idx] == '{':
        return extract_braced(s, idx)
    if s[idx] == '\\':
        m = re.match(r'^\\[a-zA-Z]+', s[idx:])
        if m:
            cmd = m.group(0)
            next_i = idx + len(cmd)
            if next_i < len(s) and s[next_i] == '{':
                inner, next_i2 = extract_braced(s, next_i)
                return f"{cmd}{{{inner}}}", next_i2
            return cmd, next_i
    return s[idx], idx + 1

def preprocess_latex(s):
    s = s.strip()
    # Replace \mathbf{X} -> X, \boldsymbol{\mu} -> \mu, \text{...} -> ...
    while True:
        m = re.search(r'\\(mathbf|boldsymbol|mathcal|mathbb|text)\{([^{}]*)\}', s)
        if not m:
            break
        kind, inner = m.group(1), m.group(2)
        if kind == 'mathbb' and inner == 'R':
            rep = 'ℝ'
        elif kind == 'mathbb' and inner == 'I':
            rep = '𝕀'
        else:
            rep = inner
        s = s[:m.start()] + rep + s[m.end():]
    # Replace longer commands first
    for k in sorted(LATEX_SYMBOLS.keys(), key=len, reverse=True):
        s = s.replace(k, LATEX_SYMBOLS[k])
    return s

def make_omml_run(text, plain=True):
    if not text:
        return ""
    text = text.replace('⦃', '{').replace('⦄', '}')
    esc = html.escape(text)
    if plain:
        return f'<m:r><m:rPr><m:sty m:val="p"/></m:rPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/><w:color w:val="000000"/></w:rPr><m:t>{esc}</m:t></m:r>'
    else:
        return f'<m:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/><w:color w:val="000000"/></w:rPr><m:t>{esc}</m:t></m:r>'

def latex_to_omml_nodes(s):
    """Convert a LaTeX math string into inner OMML XML string (<m:r>, <m:f>, <m:sSub>, etc.)."""
    s = preprocess_latex(s)
    out = []
    i = 0
    buf = ""

    def flush_buf():
        nonlocal buf
        if buf:
            out.append(make_omml_run(buf, plain=False))
            buf = ""

    while i < len(s):
        # Check \frac{num}{den}
        if s.startswith(r'\frac', i):
            flush_buf()
            i += 5
            while i < len(s) and s[i] == ' ':
                i += 1
            num_str, i = extract_atom(s, i)
            while i < len(s) and s[i] == ' ':
                i += 1
            den_str, i = extract_atom(s, i)
            num_xml = latex_to_omml_nodes(num_str)
            den_xml = latex_to_omml_nodes(den_str)
            out.append(f'<m:f><m:num>{num_xml}</m:num><m:den>{den_xml}</m:den></m:f>')
            continue

        # Check \sqrt{body}
        if s.startswith(r'\sqrt', i):
            flush_buf()
            i += 5
            while i < len(s) and s[i] == ' ':
                i += 1
            body_str, i = extract_atom(s, i)
            body_xml = latex_to_omml_nodes(body_str)
            out.append(f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{body_xml}</m:e></m:rad>')
            continue

        # Check \sum_{sub}^{sup}
        if s.startswith(r'\sum', i):
            flush_buf()
            i += 4
            sub_str, sup_str = "", ""
            while i < len(s) and s[i] in (' ', '_', '^'):
                if s[i] == ' ':
                    i += 1
                elif s[i] == '_':
                    sub_str, i = extract_atom(s, i + 1)
                elif s[i] == '^':
                    sup_str, i = extract_atom(s, i + 1)
            sub_xml = latex_to_omml_nodes(sub_str) if sub_str else make_omml_run("")
            sup_xml = latex_to_omml_nodes(sup_str) if sup_str else make_omml_run("")
            # Remainder of term or space-delimited operand
            out.append(
                f'<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>'
                f'<m:sub>{sub_xml}</m:sub><m:sup>{sup_xml}</m:sup><m:e>{make_omml_run(" ")}</m:e></m:nary>'
            )
            continue

        # Check base followed by _ or ^
        if i + 1 < len(s) and s[i] not in ('_', '^', ' ', '{', '}') and s[i+1] in ('_', '^'):
            # flush previous buffer except current char
            flush_buf()
            base_char = s[i]
            i += 1
            sub_str, sup_str = None, None
            while i < len(s) and s[i] in ('_', '^'):
                if s[i] == '_':
                    sub_str, i = extract_atom(s, i + 1)
                elif s[i] == '^':
                    sup_str, i = extract_atom(s, i + 1)
            base_xml = make_omml_run(base_char, plain=False)
            if sub_str is not None and sup_str is not None:
                out.append(f'<m:sSubSup><m:e>{base_xml}</m:e><m:sub>{latex_to_omml_nodes(sub_str)}</m:sub><m:sup>{latex_to_omml_nodes(sup_str)}</m:sup></m:sSubSup>')
            elif sub_str is not None:
                out.append(f'<m:sSub><m:e>{base_xml}</m:e><m:sub>{latex_to_omml_nodes(sub_str)}</m:sub></m:sSub>')
            elif sup_str is not None:
                out.append(f'<m:sSup><m:e>{base_xml}</m:e><m:sup>{latex_to_omml_nodes(sup_str)}</m:sup></m:sSup>')
            continue

        # Check multi-letter word followed by _ or ^ (e.g. max_{...}, Output_i, Quantile_{0.85})
        if s[i] in ('_', '^') and buf:
            # Split last word from buf as base
            m_word = re.search(r'([a-zA-Z0-9‖\)]+)$', buf)
            if m_word:
                base_word = m_word.group(1)
                buf = buf[:m_word.start()]
                flush_buf()
            else:
                base_word = buf[-1]
                buf = buf[:-1]
                flush_buf()
            sub_str, sup_str = None, None
            while i < len(s) and s[i] in ('_', '^'):
                if s[i] == '_':
                    sub_str, i = extract_atom(s, i + 1)
                elif s[i] == '^':
                    sup_str, i = extract_atom(s, i + 1)
            base_xml = make_omml_run(base_word, plain=(len(base_word) > 1))
            if sub_str is not None and sup_str is not None:
                out.append(f'<m:sSubSup><m:e>{base_xml}</m:e><m:sub>{latex_to_omml_nodes(sub_str)}</m:sub><m:sup>{latex_to_omml_nodes(sup_str)}</m:sup></m:sSubSup>')
            elif sub_str is not None:
                out.append(f'<m:sSub><m:e>{base_xml}</m:e><m:sub>{latex_to_omml_nodes(sub_str)}</m:sub></m:sSub>')
            elif sup_str is not None:
                out.append(f'<m:sSup><m:e>{base_xml}</m:e><m:sup>{latex_to_omml_nodes(sup_str)}</m:sup></m:sSup>')
            continue

        if s[i] in ('{', '}'):
            i += 1
            continue

        buf += s[i]
        i += 1

    flush_buf()
    return "".join(out)

def append_display_equation(doc, latex_str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    inner_omml = latex_to_omml_nodes(latex_str)
    omml_xml = f'<m:oMathPara {MATH_NS}><m:oMath>{inner_omml}</m:oMath></m:oMathPara>'
    try:
        p._p.append(parse_xml(omml_xml))
    except Exception as e:
        # Fallback to clean formatted run if XML error
        run = p.add_run(preprocess_latex(latex_str))
        set_run_font_strict(run, "Times New Roman", 11.5, italic=True, color_rgb=(0, 0, 0))

def append_inline_math_runs(p, math_str, font_size=12, base_bold=False, color_rgb=None):
    """Parse an inline math string like 'f_0 \\approx 32.7\\text{ Hz}' into clean Times New Roman runs with subscript/superscript."""
    s = preprocess_latex(math_str).replace('⦃', '{').replace('⦄', '}')
    i = 0
    while i < len(s):
        if s[i] in ('_', '^'):
            mode = 'sub' if s[i] == '_' else 'sup'
            atom, i = extract_atom(s, i + 1)
            # Clean nested subscripts inside superscripts e.g. T_q -> Tq, N_{gallery} -> N_gallery
            atom_clean = re.sub(r'_\{([^{}]*)\}', r'(\1)', atom)
            atom_clean = re.sub(r'_([a-zA-Z0-9]+)', r'\1', atom_clean)
            atom_clean = atom_clean.replace('{', '').replace('}', '')
            r = p.add_run(atom_clean)
            set_run_font_strict(
                r, "Times New Roman", font_size,
                bold=base_bold, italic=False,
                subscript=(mode == 'sub'), superscript=(mode == 'sup'),
                color_rgb=color_rgb
            )
        elif s[i] in ('{', '}'):
            i += 1
        else:
            # Read until next _ or ^ or { or }
            j = i
            while j < len(s) and s[j] not in ('_', '^', '{', '}'):
                j += 1
            chunk = s[i:j]
            # If chunk is a single math variable letter, italicize it
            is_var = bool(re.match(r'^[a-zA-Z]$', chunk.strip()))
            r = p.add_run(chunk)
            set_run_font_strict(
                r, "Times New Roman", font_size,
                bold=base_bold, italic=is_var,
                color_rgb=color_rgb
            )
            i = j

def clean_markdown_links_and_icons(text):
    # Replace [label](url) -> label
    text = re.sub(r'\[([^\[\]]+)\]\([^()\s]+\)', r'\1', text)
    # Remove emoji icons that break Times New Roman
    for icon in ['📁', '📄', '📝', '📂', '👉', '☁️', '📓', '📦', '📕', '📘', '📗', '✅', '⏳', '★']:
        text = text.replace(icon, '')
    return text

def append_rich_text(p, text, font_size=12, default_bold=False, default_italic=False, color_rgb=(0, 0, 0)):
    """Parse **bold**, *italic*, `code`, and $inline_math$ into clean Times New Roman runs in pure black."""
    text = clean_markdown_links_and_icons(text)
    color_rgb = (0, 0, 0)
    # Tokenize by **...**, *...*, `...`, $...$
    pattern = re.compile(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)')
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            plain = text[pos:m.start()]
            r = p.add_run(plain)
            set_run_font_strict(r, "Times New Roman", font_size, bold=default_bold, italic=default_italic, color_rgb=color_rgb)
        token = m.group(0)
        if token.startswith('**') and token.endswith('**'):
            inner = token[2:-2]
            # Inner might contain $...$
            if '$' in inner:
                append_rich_text(p, inner, font_size=font_size, default_bold=True, default_italic=default_italic, color_rgb=color_rgb)
            else:
                r = p.add_run(inner)
                set_run_font_strict(r, "Times New Roman", font_size, bold=True, italic=default_italic, color_rgb=color_rgb)
        elif token.startswith('*') and token.endswith('*'):
            inner = token[1:-1]
            r = p.add_run(inner)
            set_run_font_strict(r, "Times New Roman", font_size, bold=default_bold, italic=True, color_rgb=color_rgb)
        elif token.startswith('`') and token.endswith('`'):
            inner = token[1:-1]
            r = p.add_run(inner)
            set_run_font_strict(r, "Times New Roman", font_size, bold=True, italic=False, color_rgb=(0, 0, 0))
        elif token.startswith('$') and token.endswith('$'):
            inner = token[1:-1]
            append_inline_math_runs(p, inner, font_size=font_size, base_bold=default_bold, color_rgb=(0, 0, 0))
        pos = m.end()
    if pos < len(text):
        r = p.add_run(text[pos:])
        set_run_font_strict(r, "Times New Roman", font_size, bold=default_bold, italic=default_italic, color_rgb=color_rgb)

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="CBD5E0", sz="4"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def merge_master_markdown(md_path):
    chapters = [f'docs/report/chuong_{i}.md' for i in range(1, 8)]
    front_matter = '''# BÁO CÁO DỰ ÁN CHUYÊN NGÀNH 1 (PBL6)
## ĐỀ TÀI: HỆ THỐNG TRUY VẤN BÀI HÁT QUA TIẾNG NGÂN NGA (QUERY-BY-HUMMING) DỰA TRÊN CONFORMER-BYTEHUM V6.2

---

## TÓM TẮT DỰ ÁN (ABSTRACT)

Báo cáo trình bày toàn diện quá trình nghiên cứu, thiết kế kiến trúc học sâu, tối ưu hóa xử lý tín hiệu và triển khai thực tế của Hệ thống Truy vấn Bài hát qua Tiếng ngân nga (Query-by-Humming - QBH) phiên bản Conformer-ByteHum V6.2. 

Khác biệt với các hệ thống truyền thống chỉ dựa trên mạng CNN nông và phổ Chroma CQT 12 bins gập quãng tám (như khảo sát trong QBH.pdf), dự án đề xuất giải pháp đột phá kết hợp 4 trụ cột công nghệ:
1. **Xử lý tín hiệu cao cấp**: Sử dụng Meta HTDemucs v4 bóc tách 100% Vocal Stems sạch từ bản Master Studio, kết hợp phép biến đổi Full CQT 96 bins (8 quãng tám) và chuỗi Data Augmentation đặc thù (Pitch Shift $\\pm 12$ bán âm, Reverb, Harmonic Noise).
2. **Kiến trúc mô hình Conformer-ByteHum 9.91M tham số**: Kết hợp khả năng nắm bắt nhịp điệu cục bộ của Depthwise Convolution và ngữ cảnh giai điệu toàn cục của Multi-Head Self-Attention (MHSA). Tích hợp cơ chế Attentive Statistics Pooling (ASP) cùng bộ ba mô đun tinh hoa của ByteHum (ICASSP 2024): LAT (Learned Alignment Transformer) với MaxMean Pooling, LAC (Learned Alignment Classifier) với L2 Prototypes, và GRL (Gradient Reversal Layer) huấn luyện đối kháng miền.
3. **Thuật toán truy hồi hai chặng SOTA**: Chặng 1 lọc ứng viên diện rộng bằng Local Frame Quantile Pooling Q85 kết hợp bộ lọc Tích chập Đường chéo Đa nhịp độ (Multi-Tempo Diagonal Shingles Conv2D quét 3 nhịp độ 0.8x, 1.0x, 1.25x) và Multi-Key CQT Pitch Shift TTA, mở rộng shortlist lên Top-50 ứng viên (đạt độ phủ 83.71% bài hát đúng). Chặng 2 thực hiện tái xếp hạng chuyên sâu bằng Subsequence DTW Chroma/Melody và Reciprocal Rank Fusion (RRF).
4. **Hạ tầng phân phối đám mây Cloudinary CDN**: Lưu trữ và cung cấp luồng phát trực tiếp (Direct Streaming URL) cho 300 bài hát với độ trễ tìm kiếm toàn trình chỉ $315\\text{ ms}$ trên GPU.

Kết quả thực nghiệm trên tập kiểm thử độc lập 1,467 queries khẳng định sự vượt trội toàn diện: chỉ số **MRR bứt phá lên 0.5489 (54.89%)** (tăng +45.7% so với CNN cơ sở và +12.2% so với V6.1), **Recall@1 đạt 45.60%** (tăng gấp 2.73 lần), **Recall@5 đạt 64.83%**, **Recall@10 đạt 71.71%**, **Recall@20 đạt 77.85%** và **Recall@50 đạt 83.09%**. Tỷ số thắng thua đối đầu trực diện đạt 2.67 : 1.

---

## DANH MỤC TỪ VIẾT TẮT VÀ THUẬT NGỮ

| Từ viết tắt | Thuật ngữ tiếng Anh đầy đủ | Diễn giải ý nghĩa |
| :--- | :--- | :--- |
| **QBH** | Query-by-Humming | Hệ thống truy vấn bài hát qua tiếng ngân nga, hát chay hoặc huýt sáo |
| **MIR** | Music Information Retrieval | Lĩnh vực khoa học công nghệ Truy xuất Thông tin Âm nhạc |
| **CQT** | Constant-Q Transform | Biến đổi Q hằng số với thang tần số logarit |
| **STFT** | Short-Time Fourier Transform | Biến đổi Fourier thời gian ngắn với dải tần số tuyến tính |
| **MHSA** | Multi-Head Self-Attention | Cơ chế Tự chú ý đa đầu trong kiến trúc Transformer |
| **ASP** | Attentive Statistics Pooling | Khối gộp thống kê kỳ vọng và độ lệch chuẩn có trọng số tự chú ý |
| **LAT** | Learned Alignment Transformer | Bộ căn chỉnh giai điệu cục bộ thông qua Cross-Attention MaxMean |
| **LAC** | Learned Alignment Classifier | Bộ phân loại căn chỉnh dựa trên vector tâm cụm nguyên mẫu L2 |
| **GRL** | Gradient Reversal Layer | Lớp đảo ngược dấu gradient phục vụ huấn luyện đối kháng miền |
| **MRR** | Mean Reciprocal Rank | Xếp hạng nghịch đảo trung bình của bài hát đúng |
| **CDN** | Content Delivery Network | Mạng lưới phân phối nội dung đa phương tiện toàn cầu (Cloudinary) |
| **RMS** | Root Mean Square | Giá trị căn bậc hai trung bình bình phương (đo năng lượng âm thanh) |
| **BPM** | Beats Per Minute | Số phách nhịp trong mỗi phút (tốc độ bài hát) |

---
'''
    merged_content = [front_matter]
    for ch_path in chapters:
        with open(ch_path, 'r', encoding='utf-8') as f:
            merged_content.append(f.read())
            merged_content.append('\n\n---\n\n')

    full_text = '\n'.join(merged_content)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"Merged master markdown: {md_path} ({len(full_text.encode('utf-8'))} bytes)")

def create_report_docx(md_path, docx_path):
    merge_master_markdown(md_path)
    
    doc = docx.Document()
    
    # Page Setup: Vietnamese Academic Thesis Standard (Top 2cm, Bottom 2cm, Left 3cm, Right 2cm)
    for section in doc.sections:
        section.top_margin = Inches(0.79)     # 2.0 cm
        section.bottom_margin = Inches(0.79)  # 2.0 cm
        section.left_margin = Inches(1.18)    # 3.0 cm
        section.right_margin = Inches(0.79)   # 2.0 cm
        section.different_first_page_header_footer = True
        
        footer = section.footer
        p_ft = footer.paragraphs[0]
        p_ft.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_ft_run = p_ft.add_run("Báo cáo Dự án Chuyên ngành 1 (PBL6): Hệ thống Conformer-ByteHum QbH V6.2")
        set_run_font_strict(p_ft_run, "Times New Roman", 9.5, italic=True, color_rgb=(0, 0, 0))

    # Strictly enforce Times New Roman across all built-in styles
    for style_name in ['Normal', 'List Bullet', 'List Number', 'Heading 1', 'Heading 2', 'Heading 3']:
        if style_name in doc.styles:
            set_style_font_strict(doc.styles[style_name], "Times New Roman", 12, color_rgb=(0, 0, 0))

    normal_style = doc.styles['Normal']
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(4)

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_table = False
    table_lines = []
    in_code_block = False
    code_lines = []

    def flush_table(lines):
        if not lines:
            return
        parsed_rows = []
        for line in lines:
            if not line.strip() or re.match(r'^\|?\s*:?-+:?\s*\|', line):
                continue
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells:
                parsed_rows.append(cells)
        if not parsed_rows:
            return
        
        num_cols = max(len(row) for row in parsed_rows)
        tbl = doc.add_table(rows=len(parsed_rows), cols=num_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl, color="000000", sz="4")
        
        for r_idx, row in enumerate(parsed_rows):
            trPr = tbl.rows[r_idx]._tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            if r_idx == 0:
                trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            for c_idx, val in enumerate(row):
                if c_idx < num_cols:
                    cell = tbl.cell(r_idx, c_idx)
                    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                    set_cell_margins(cell, top=110, bottom=110, left=140, right=140)
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.15
                    if r_idx == 0:
                        set_cell_background(cell, "EAEAEA")
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        append_rich_text(p, val, font_size=10.5, default_bold=True, color_rgb=(0, 0, 0))
                    else:
                        set_cell_background(cell, "FFFFFF")
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        append_rich_text(p, val, font_size=10.5, default_bold=(c_idx == 0), color_rgb=(0, 0, 0))
        doc.add_paragraph()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Code block handler (for source code & terminal logs wrapped in a bordered frame)
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                # Render inside a clean single-cell bordered frame in Times New Roman black (no spellcheck squiggles)
                tbl_code = doc.add_table(rows=1, cols=1)
                tbl_code.alignment = WD_TABLE_ALIGNMENT.CENTER
                set_table_borders(tbl_code, color="000000", sz="6")
                c_cell = tbl_code.cell(0, 0)
                set_cell_background(c_cell, "F8F8F8")
                set_cell_margins(c_cell, top=120, bottom=120, left=160, right=160)
                p_code = c_cell.paragraphs[0]
                p_code.paragraph_format.space_before = Pt(0)
                p_code.paragraph_format.space_after = Pt(0)
                p_code.paragraph_format.line_spacing = 1.15
                p_code.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for idx_c, c_ln in enumerate(code_lines):
                    # Replace leading spaces with non-breaking spaces so indentation is preserved in Times New Roman
                    leading_spaces = len(c_ln) - len(c_ln.lstrip(' '))
                    formatted_ln = ('\u00A0' * leading_spaces) + c_ln.lstrip(' ')
                    if idx_c > 0:
                        formatted_ln = '\n' + formatted_ln
                    run_c = p_code.add_run(formatted_ln)
                    set_run_font_strict(run_c, "Times New Roman", 10.0, color_rgb=(0, 0, 0))
                    rPr_c = run_c._r.get_or_add_rPr()
                    rPr_c.append(parse_xml(f'<w:noProof {nsdecls("w")}/>'))
                doc.add_paragraph()
                code_lines = []
            else:
                if in_table:
                    flush_table(table_lines)
                    in_table = False
                    table_lines = []
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Table handler
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                flush_table(table_lines)
                in_table = False
                table_lines = []

        if not line.strip():
            i += 1
            continue

        if re.match(r'^---+$', line.strip()):
            i += 1
            continue

        # Display Equation Handler: $$...$$
        if line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 4:
            eq_str = line.strip()[2:-2].strip()
            append_display_equation(doc, eq_str)
            i += 1
            continue

        # Image Handler: ![alt](path)
        img_match = re.match(r'^!\[(.*?)\]\((.*?)\)', line.strip())
        if img_match:
            img_path = img_match.group(2).strip()
            if os.path.exists(img_path):
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(8)
                p_img.paragraph_format.space_after = Pt(2)
                p_img.paragraph_format.keep_with_next = True
                p_img.add_run().add_picture(img_path, width=Inches(6.2))
            i += 1
            continue

        # Figure Caption: *Hình x.x: ...*
        if line.strip().startswith("*Hình ") and line.strip().endswith("*"):
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(10)
            append_rich_text(p_cap, line.strip()[1:-1], font_size=11, default_italic=True, color_rgb=(0, 0, 0))
            i += 1
            continue

        # Headings
        if line.startswith("# "):
            title_text = line[2:].strip()
            if "CHƯƠNG" in title_text.upper() or "TÀI LIỆU THAM KHẢO" in title_text.upper() or "PHỤ LỤC" in title_text.upper():
                doc.add_page_break()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_with_next = True
            if "BÁO CÁO" in title_text:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            append_rich_text(p, title_text, font_size=15, default_bold=True, color_rgb=(0, 0, 0))
            i += 1
            continue

        if line.startswith("## "):
            h2_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            if "ĐỀ TÀI" in h2_text:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            append_rich_text(p, h2_text, font_size=13.5, default_bold=True, color_rgb=(0, 0, 0))
            i += 1
            continue

        if line.startswith("### "):
            h3_text = line[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            append_rich_text(p, h3_text, font_size=12.5, default_bold=True, color_rgb=(0, 0, 0))
            i += 1
            continue

        # Bullet list items
        if re.match(r'^\s*[-*]\s+', line):
            indent_level = len(re.match(r'^\s*', line).group(0)) // 2
            item_text = re.sub(r'^\s*[-*]\s+', '', line)
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.left_indent = Inches(0.28 * (indent_level + 1))
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.25
            append_rich_text(p, item_text, font_size=12, color_rgb=(0, 0, 0))
            i += 1
            continue

        # Numbered list items (explicit numbering so each section starts at 1.)
        num_match = re.match(r'^(\s*)(\d+\.)\s+(.*)$', line)
        if num_match:
            indent_spaces, num_label, item_text = num_match.group(1), num_match.group(2), num_match.group(3)
            indent_level = len(indent_spaces) // 2
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.28 * (indent_level + 1))
            p.paragraph_format.first_line_indent = Inches(-0.22)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.25
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r_num = p.add_run(f"{num_label} ")
            set_run_font_strict(r_num, "Times New Roman", 12, bold=True, color_rgb=(0, 0, 0))
            append_rich_text(p, item_text, font_size=12, color_rgb=(0, 0, 0))
            i += 1
            continue

        # Regular Paragraph
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_after = Pt(5)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        append_rich_text(p, line, font_size=12, color_rgb=(0, 0, 0))
        i += 1

    if in_table:
        flush_table(table_lines)

    os.makedirs(os.path.dirname(docx_path), exist_ok=True)
    doc.save(docx_path)
    print(f"Successfully exported DOCX to {docx_path}, size: {os.path.getsize(docx_path)} bytes")

if __name__ == '__main__':
    create_report_docx('docs/report/BAO_CAO_PBL6_QBH_FINAL.md', 'docs/report/BAO_CAO_PBL6_QBH_FINAL.docx')

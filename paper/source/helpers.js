const fs = require('fs');
const d = require('docx');
const { Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  AlignmentType, HeadingLevel, TabStopType, Math: OMath, MathRun, MathFraction, MathSum, MathSubScript,
  MathSuperScript, MathRadical, MathSubSuperScript } = d;

const FONT = 'Times New Roman';
// A4 (11906 DXA) with left margin 1.5" (2160), top/bottom/right 1" (1440)
const CONTENT_W = 11906 - 2160 - 1440; // = 8306
const SINGLE = { line: 240, lineRule: 'auto' };   // single spacing, used inside tables/captions/refs
function scaleW(widths) {   // rescale a width array (designed for any total) onto CONTENT_W
  const s = widths.reduce((a, b) => a + b, 0);
  const out = widths.map((w) => Math.round((w * CONTENT_W) / s));
  out[out.length - 1] += CONTENT_W - out.reduce((a, b) => a + b, 0); // fix rounding
  return out;
}

// inline markup: **bold**, _italic_, ^sup^, ~sub~
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|(?<![A-Za-z0-9])_[^_]+_(?![A-Za-z0-9])|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith('_')) out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else if (t.startsWith('^')) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), subScript: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}), alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: 0, before: opts.before ?? 0 }, indent: opts.indent,
  keepNext: opts.keepNext,
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], keepNext: true,
  spacing: { after: 0, before: 260 } });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], keepNext: true,
  spacing: { after: 0, before: 190 } });
const BUL = (text) => new Paragraph({ numbering: { reference: 'bullets', level: 0 }, children: runs(text),
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 0, line: 240, lineRule: 'auto' } });
const NUM = (text, ref = 'nums') => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text),
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 0, line: 240, lineRule: 'auto' } });

function caption(label, text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 100, ...SINGLE },
    children: [new TextRun({ text: label + ' ', bold: true, size: 17 }), ...runs(text, { size: 17 })] });
}
function tcaption(label, text) {   // table captions go above the table
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40, ...SINGLE }, keepNext: true,
    children: [new TextRun({ text: label + ' ', bold: true, size: 17 }), ...runs(text, { size: 17 })] });
}
function figure(path, widthIn, label, text) {
  const buf = fs.readFileSync(path);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);   // PNG header
  const W = Math.round(widthIn * 96), H = Math.round(W * h / w);
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 60, after: 0, ...SINGLE },
    children: [new ImageRun({ type: 'png', data: buf, transformation: { width: W, height: H } })] }),
    caption(label, text)];
}

const thin = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
// academic three-rule table; rows[0] = header; widths in DXA; bold rows by index
function table(rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const n = rows.length;
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        margins: { top: 55, bottom: 55, left: 90, right: 90 },
        shading: (opts.shadeRows || []).includes(i) ? { type: ShadingType.CLEAR, fill: 'EEF3FA', color: 'auto' } : undefined,
        borders: { left: none, right: none,
          top: i === 0 ? thin : (opts.ruleAbove || []).includes(i) ? thin : none,
          bottom: i === 0 || i === n - 1 ? thin : none },
        children: [new Paragraph({ alignment: j === 0 && !opts.centerFirst ? AlignmentType.LEFT : AlignmentType.CENTER,
          spacing: { after: 0, ...SINGLE }, keepNext: opts.keep ? i < n - 1 : undefined,
          children: runs(String(c), { size: opts.size || 18, bold: i === 0 || (opts.boldRows || []).includes(i) }) })],
      })) })),
  });
}

// equation paragraph: centered math with right-aligned number
function eq(mathChildren, num) {
  return new Paragraph({ tabStops: [{ type: TabStopType.CENTER, position: CONTENT_W / 2 }, { type: TabStopType.RIGHT, position: CONTENT_W }],
    spacing: { before: 60, after: 100 },
    children: [new TextRun('\t'), new OMath({ children: mathChildren }), new TextRun(`\t(${num})`)] });
}
function eqImg(path, num) {
  const buf = fs.readFileSync(path);
  const w = buf.readUInt32BE(16), hh = buf.readUInt32BE(20);
  const W = Math.round(w * 96 / 300), Hh = Math.round(hh * 96 / 300);   // rendered at 300 dpi
  return new Paragraph({ tabStops: [{ type: TabStopType.CENTER, position: CONTENT_W / 2 }, { type: TabStopType.RIGHT, position: CONTENT_W }],
    spacing: { before: 40, after: 0, ...SINGLE },
    children: [new TextRun('\t'), new ImageRun({ type: 'png', data: buf, transformation: { width: W, height: Hh } }), new TextRun(`\t(${num})`)] });
}
const mr = (t) => new MathRun(t);
const frac = (n, dd) => new MathFraction({ numerator: n, denominator: dd });
const sub = (b, s) => new MathSubScript({ children: [mr(b)], subScript: [mr(s)] });
const sup = (b, s) => new MathSuperScript({ children: Array.isArray(b) ? b : [mr(b)], superScript: Array.isArray(s) ? s : [mr(s)] });
const sum = (below, above, body) => new MathSum({ children: body, subScript: [mr(below)], superScript: above ? [mr(above)] : undefined });
const sqrt = (body) => new MathRadical({ children: body });

module.exports = { eqImg, d, FONT, CONTENT_W, SINGLE, scaleW, runs, P, H1, H2, BUL, NUM, caption, tcaption, figure, table, eq, mr, frac, sub, sup, sum, sqrt };

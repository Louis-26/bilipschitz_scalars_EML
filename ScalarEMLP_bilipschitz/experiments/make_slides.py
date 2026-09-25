# usage: pip install python-pptx; python experiments/make_slides.py deck_from_google.pptx ../output/bilipschitz_capstone_results.pptx ../output/figures
# (input: File > Download > pptx of the Google Slides deck; figures from experiments/make_figures.py)
"""fill the capstone deck (Google Slides export) with the stage-1 results; keeps the original theme and slides"""
import copy
import sys

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

SRC, OUT, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
INNER, BILIP, INK, INK2, TINT = (RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xEB, 0x68, 0x34), RGBColor(0x0B, 0x0B, 0x0B),
                                 RGBColor(0x52, 0x51, 0x4E), RGBColor(0xF3, 0xF3, 0xF1))
FONT = "Arial"
prs = Presentation(SRC)
L_BODY, L_TITLE_ONLY = prs.slide_layouts[2], prs.slide_layouts[4]


def text(slide, x, y, w, h, paras, size=14, color=INK, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, space=6):
    """paras: list of str or list of (text, {bold, color, size, italic}) runs; a str starting with '• ' is a bullet"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, p in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.space_after = Pt(space)
        runs = [(p, {})] if isinstance(p, str) else p
        bullet = isinstance(p, str) and p.startswith("• ")
        if bullet:
            runs = [(p[2:], {})]
            pPr = par._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.2))); pPr.set("indent", str(-Inches(0.2)))
            bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
            pPr.append(bu)
        for t, st in runs:
            r = par.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(st.get("size", size))
            f.bold = st.get("bold", False)
            f.italic = st.get("italic", False)
            f.color.rgb = st.get("color", color)
    return tb


def picture(slide, name, x, y, w, h):
    """place a figure fitted into the box (x, y, w, h) keeping its aspect ratio, centred horizontally;
    returns the bottom edge (inches) so captions can sit right below it"""
    path = f"{FIG}/{name}"
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y), Inches(pw), Inches(ph))
    return y + ph


def card(slide, x, y, w, h):
    shp = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))  # rectangle
    shp.fill.solid(); shp.fill.fore_color.rgb = TINT
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def stat(slide, x, y, w, big, color, label):
    card(slide, x, y, w, 1.12)
    text(slide, x + 0.18, y + 0.1, w - 0.36, 0.55, [[(big, {"bold": True, "color": color, "size": 30})]])
    text(slide, x + 0.18, y + 0.66, w - 0.36, 0.42, [label], size=11, color=INK2)


def set_title(slide, t):
    slide.shapes.title.text_frame.text = t


def body_placeholder(slide):
    for sh in slide.placeholders:
        if sh.placeholder_format.idx == 1:
            return sh


def remove(shape):
    shape._element.getparent().remove(shape._element)


slides = list(prs.slides)

# ---- slide 2: complete the three stages (keep the original runs and style)
s2 = slides[1]
b = body_placeholder(s2)
add = {"Stage 1": "Yao et al. (2021): scalar-based HNN and Neural ODE on the springy double pendulum",
       "Stage 2": "i.e. replace the Gram matrix XXᵀ of the scalar features by its square root (XXᵀ)^(1/2), then tune hyperparameters",
       "Stage 3": ": Gaussian noise on the training data and on test initial conditions (adversarial attack: next step)"}
for par in b.text_frame.paragraphs:
    t = "".join(r.text for r in par.runs)
    for k, v in add.items():
        if t.startswith(k) and par.runs:
            last = par.runs[-1]
            if v.startswith(":"):
                last.text = last.text.rstrip()
            elif not last.text.endswith(" "):
                last.text += " "
            r = copy.deepcopy(last._r)
            last._r.addnext(r)
            par.runs[-1].text = v
s2.notes_slide.notes_text_frame.text = (
    "Three stages. Stage 1 reproduces Table 1 of Yao et al. 2021 with the upstream code. Stage 2 swaps the inner-"
    "product (Gram) block of the 30 scalar features for its matrix square root, which is bilipschitz, and tunes "
    "the models. Stage 3 adds Gaussian noise; the adversarial attack is the next step.")

# ---- slide 3: problem setting (reuse the empty slide)
s3 = slides[2]
set_title(s3, "Problem setting: learning the springy double pendulum")
remove(body_placeholder(s3))
text(s3, 0.5, 1.3, 4.6, 3.6, [
    "• State z = (q₁, q₂, p₁, p₂) ∈ (ℝ³)⁴, gravity g = (0, 0, −1); all masses, stiffnesses and lengths = 1",
    "• 5000 simulated trajectories; train / val / test = 500 / 500 / 500 chunks of 5 steps (Δt = 0.2)",
    "• HNN: MLP → H(z), dynamics J∇H.  N-ODE: MLP → 24 invariant coefficients of an equivariant field",
    "• Metric: geometric mean of the state relative error over 150-step rollouts (paper Table 1)",
], size=13, space=10)
rows = [("30 invariant scalars", "#", True), ("norms ‖uᵢ‖", "4", False), ("Gram uᵢᵀuⱼ  →  bilipschitz block", "16", False),
        ("gravity gᵀuᵢ", "4", False), ("spring 2: |q₁−q₂|², |q₁−q₂|", "2", False), ("(q₁−q₂)ᵀuᵢ", "4", False)]
tbl = s3.shapes.add_table(len(rows), 2, Inches(5.5), Inches(1.35), Inches(4.0), Inches(2.6)).table
tbl.columns[0].width = Inches(3.35); tbl.columns[1].width = Inches(0.65)
for i, (a, n, head) in enumerate(rows):
    for j, v in enumerate([a, n]):
        c = tbl.cell(i, j)
        c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if not head else TINT
        tf = c.text_frame; tf.text = ""
        r = tf.paragraphs[0].add_run(); r.text = v
        r.font.name = FONT; r.font.size = Pt(12); r.font.bold = head or i == 2
        r.font.color.rgb = BILIP if (i == 2) else INK
        tf.paragraphs[0].alignment = PP_ALIGN.RIGHT if j == 1 else PP_ALIGN.LEFT
text(s3, 5.5, 4.1, 4.0, 0.5, ["u = (q₁, q₂, p₁, p₂). Only the orange block changes between the two models."],
     size=10, color=INK2)
s3.notes_slide.notes_text_frame.text = (
    "The task and data are exactly those of Yao et al. 2021. The scalar method feeds 30 O(3)-invariant scalars "
    "to an MLP. The 16 Gram entries are the only features that the bilipschitz embedder replaces, so the "
    "comparison isolates the embedding.")

# ---- slide 4: the embedder
s4 = slides[3]
set_title(s4, "Bilipschitz embedder: θ(X) = (XXᵀ)^(1/2)")
remove(body_placeholder(s4))
text(s4, 0.5, 1.3, 4.3, 3.7, [
    [("Balan & Dock (2022): ", {"bold": True}),
     ("D(X,Y) ≤ ‖θ(X) − θ(Y)‖ ≤ √2 · D(X,Y),  D = minᵤ ‖X − YU‖ (Procrustes distance)", {})],
    [("Gram XXᵀ has no such bound: ", {"bold": True}), ("ratio from 1.2 to 8 on random pairs (right)", {})],
    [("Implementation: ", {"bold": True}),
     ("thin SVD X = UΣVᵀ ⇒ θ(X) = UΣUᵀ. Pure JAX, exact, twice differentiable (HNN training needs 2nd derivatives)", {})],
    [("Fixed: ", {"bold": True}),
     ("the earlier torch version crashed inside jax.grad, so no bilipschitz model had actually been trained", {})],
], size=13, space=10)
yb = picture(s4, "bilip_bound.png", 5.0, 1.35, 4.6, 3.3)
text(s4, 5.0, yb + 0.1, 4.6, 0.35, ["200 000 random + 200 000 nearby pairs X, Y ∈ ℝ⁴ˣ³: all ratios in [1.0004, 1.409]"],
     size=10, color=INK2)
s4.notes_slide.notes_text_frame.text = (
    "The square root of the Gram matrix is bilipschitz with respect to the Procrustes distance, with constants 1 "
    "and sqrt 2. The histogram checks this numerically. The 4x4 Gram matrix has rank 3, so we never take the square "
    "root of its eigenvalues directly: the thin SVD of X gives the same matrix and has finite derivatives.")

# ---- new slides
def new(title, layout=L_TITLE_ONLY):
    s = prs.slides.add_slide(layout)
    set_title(s, title)
    return s


s5 = new("Stage 1: reproducing Yao et al. (2021), Table 1")
picture(s5, "repro.png", 0.5, 1.3, 5.6, 3.6)
stat(s5, 6.5, 1.35, 3.0, "0.0104", INNER, "N-ODE, ours (paper 0.009 ± 0.001): within 1 std")
stat(s5, 6.5, 2.7, 3.0, "0.0092", INNER, "HNN, ours (paper 0.005 ± 0.002): within 2×")
text(s5, 6.5, 4.05, 3.0, 0.9, ["Upstream setting (3 × 100, lr 5e-3, 2000 epochs), 3 seeds. Needed: restoring the "
                                "upstream Hamiltonian (k₂ instead of ½k₂) and regenerating the data."],
     size=10, color=INK2)
s5.notes_slide.notes_text_frame.text = (
    "With the upstream Hamiltonian and hyperparameters, the N-ODE matches the paper within one standard deviation "
    "and the HNN is within a factor 2. The branch had a modified Hamiltonian, so all older caches were regenerated.")

s6 = new("Stage 2: the HNN needs a smooth activation")
picture(s6, "relu_vs_smooth.png", 0.5, 1.3, 5.6, 3.6)
text(s6, 6.4, 1.35, 3.2, 3.6, [
    [("ReLU: ", {"bold": True}), ("H is piecewise linear in the scalars, so the vector field J∇H is piecewise constant", {})],
    [("Inner product: ", {"bold": True}), ("the true H is exactly linear in the scalars (½|p|² is a Gram entry), ReLU is harmless", {})],
    [("Square root: ", {"bold": True}), ("|p|² becomes quadratic → loss plateau, rollout error 0.32 for every size / lr", {})],
    [("SiLU / softplus: ", {"bold": True, "color": BILIP}), ("0.32 → 0.0074 (43× better)", {"color": BILIP, "bold": True})],
], size=12, space=10)
s6.notes_slide.notes_text_frame.text = (
    "With ReLU the bilipschitz HNN never learns (dotted curve), for all 27 grid points. Gradients were checked "
    "against float64 finite differences, so this is representational: a ReLU Hamiltonian has a piecewise-constant "
    "gradient. Smooth activations fix it.")

s7 = new("Stage 2: hyperparameter search (212 runs)")
yb = picture(s7, "tuning.png", 0.5, 1.25, 9.0, 3.3)
text(s7, 0.5, yb + 0.12, 9.0, 0.5, [[
    ("Selection on the validation rollout; test numbers only reported.  ", {}),
    ("Search: ", {"bold": True}),
    ("depth {3, 5, 7} × width {100–300} × lr {3e-3 – 5e-2} × activation {ReLU, SiLU, softplus, tanh} × RBF width.  ", {}),
    ("Selected: ", {"bold": True}),
    ("HNN inner 3-100-0.03 ReLU · HNN bilip 3-200-0.02 SiLU · N-ODE inner 7-200-0.02 · N-ODE bilip 7-200-0.005", {})]],
    size=10, color=INK2)
s7.notes_slide.notes_text_frame.text = (
    "Light bars: paper setting; dark bars: selected configuration, 3 seeds. Tuning also improves the baseline: "
    "the HNN reaches 0.0015, three times better than the paper. N-ODE: the bilipschitz embedder is on par with the "
    "tuned baseline (0.0056 vs 0.0058). HNN: bilipschitz stays about 5 times worse on clean data.")

s8 = new("Stage 3: Gaussian noise on the training data")
yb = picture(s8, "noise.png", 0.5, 1.2, 9.0, 3.45)
text(s8, 0.5, yb + 0.1, 9.0, 0.45, [[
    ("N-ODE: ", {"bold": True, "color": BILIP}), ("bilipschitz is best at every noise level.   ", {}),
    ("HNN: ", {"bold": True}),
    ("the tuned ReLU inner product stays best, but the bilipschitz model degrades more slowly than the inner product "
     "with the same configuration (dotted).", {})]], size=11)
s8.notes_slide.notes_text_frame.text = (
    "Noise with standard deviation sigma times the per-coordinate scale is added to every training state; test data "
    "stay clean. Dotted lines are the control: the inner-product model trained with the hyperparameters selected "
    "for the bilipschitz model.")

s9 = new("Same hyperparameters: embedding effect alone")
picture(s9, "control.png", 0.5, 1.3, 5.9, 3.5)
stat(s9, 6.7, 1.35, 2.8, "−39 %", BILIP, "N-ODE error at σ = 0.03 (−15 to −39 % at every level)")
stat(s9, 6.7, 2.7, 2.8, "−32 %", BILIP, "HNN error at σ = 0.1 (but +23 % on clean data)")
text(s9, 6.7, 4.05, 2.8, 0.9, ["Perturbing test initial conditions instead shows no difference: that error is "
                               "dominated by the chaotic dynamics."], size=10, color=INK2)
s9.notes_slide.notes_text_frame.text = (
    "Bars: relative change of the rollout error, bilipschitz versus inner product with identical hyperparameters, "
    "3 seeds each. Apart from the -4 percent point, all differences exceed two standard deviations.")

s10 = new("Conclusions and next steps", prs.slide_layouts[4])
card(s10, 0.5, 1.3, 5.6, 3.35)
text(s10, 0.75, 1.45, 5.1, 3.4, [
    [("Findings", {"bold": True, "size": 16})],
    "• Paper reproduced: N-ODE within 1 std, HNN within 2×",
    "• N-ODE: bilipschitz scalars match the tuned baseline on clean data and cut the error by 15–39 % under training noise",
    "• HNN: needs a smooth activation; 5× worse on clean data (the true H is linear in inner products) but more robust at high noise",
    "• Initial-condition perturbations: no effect of the embedding",
], size=14, space=10)
card(s10, 6.4, 1.3, 3.1, 3.35)
text(s10, 6.65, 1.45, 2.6, 3.4, [
    [("Next steps", {"bold": True, "size": 16})],
    "• Adversarial attack on initial conditions / training data",
    "• Embeddings that keep the linear structure (add squared norms back, θ on a sub-block)",
    "• More data seeds and physical systems",
], size=14, space=10)
s10.notes_slide.notes_text_frame.text = (
    "Summary: the bilipschitz embedder helps the Neural ODE, especially with noisy data, and needs care for the HNN. "
    "Code, results and the overview folder are in the repository.")

# ---- order: 1-4, new 5-10, references
ids = prs.slides._sldIdLst
els = list(ids)
refs, rest = els[4:6], els[:4] + els[6:]
for e in els:
    ids.remove(e)
for e in rest + refs:
    ids.append(e)
prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")

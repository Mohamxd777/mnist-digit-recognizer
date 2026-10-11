import sys
import os
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette - Professional Academic Dark Tech
    BG_COLOR = RGBColor(11, 15, 25)         # #0B0F19 Deep slate navy
    CARD_BG = RGBColor(24, 32, 47)          # #18202F Slate container
    CARD_BORDER = RGBColor(46, 61, 88)      # #2E3D58 Slate border
    TEXT_WHITE = RGBColor(248, 250, 252)    # #F8FAFC Pure white
    TEXT_MUTED = RGBColor(148, 163, 184)    # #94A3B8 Slate gray
    TEXT_LIGHT = RGBColor(203, 213, 225)    # #CBD5E1 Light slate
    
    CYAN = RGBColor(56, 189, 248)           # #38BDF8 Vibrant Cyan
    INDIGO = RGBColor(129, 140, 248)        # #818CF8 Soft Indigo
    EMERALD = RGBColor(52, 211, 153)        # #34D399 Bright Emerald
    AMBER = RGBColor(251, 191, 36)          # #FBBF24 Bright Amber
    ROSE = RGBColor(251, 113, 133)          # #FB7185 Accent Rose

    def set_slide_background(slide):
        bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg_shape.fill.solid()
        bg_shape.fill.fore_color.rgb = BG_COLOR
        bg_shape.line.color.rgb = BG_COLOR
        return bg_shape

    def add_header(slide, tag_text, title_text, tag_color=CYAN):
        # Category Tag Badge
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.7), Inches(0.35))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = tag_text.upper()
        p_tag.font.size = Pt(11)
        p_tag.font.bold = True
        p_tag.font.color.rgb = tag_color
        p_tag.font.name = "Segoe UI"

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.65))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE
        p_title.font.name = "Segoe UI"

    def add_card(slide, left, top, width, height, title="", title_color=CYAN):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)

        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), Inches(0.4))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = title_color
            p.font.name = "Segoe UI"
        return card

    # =========================================================================
    # SLIDE 1: TITLE & DEFENSE OVERVIEW
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)

    # Main Hero Title Box
    h_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.3), Inches(2.2))
    tf1 = h_box.text_frame
    tf1.word_wrap = True
    
    p_badge = tf1.paragraphs[0]
    p_badge.text = "ACADEMIC DEFENSE & COURSE PROJECT | AI II / DEEP LEARNING"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = CYAN
    p_badge.font.name = "Segoe UI"
    p_badge.space_after = Pt(10)

    p_main = tf1.add_paragraph()
    p_main.text = "Handwritten Digit Recognition\n& The Deep Learning Evolution"
    p_main.font.size = Pt(36)
    p_main.font.bold = True
    p_main.font.color.rgb = TEXT_WHITE
    p_main.font.name = "Segoe UI"
    p_main.space_after = Pt(12)

    p_sub = tf1.add_paragraph()
    p_sub.text = "A Rigorous Comparative Study Across Classical ML (SVM), Deep MLP, and Modern CNN"
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = TEXT_LIGHT
    p_sub.font.name = "Segoe UI"

    # 3 Stat Cards across bottom
    stats = [
        ("21,000 Samples", "Real MNIST Test Set Evaluated", "Stratified multi-model test verification", CYAN),
        ("99.59% Top Accuracy", "Deep MLP with Batch Normalization", "Fastest neural inference (0.12 ms/sample)", EMERALD),
        ("Live Cloud Application", "Production Streamlit Deployment", "https://mnistdigitrecognizerr.streamlit.app/", INDIGO)
    ]
    card_w = Inches(3.6)
    card_h = Inches(2.3)
    start_x = Inches(1.0)
    card_y = Inches(4.2)

    for i, (metric, label, desc, color) in enumerate(stats):
        x = start_x + i * Inches(3.85)
        add_card(slide1, x, card_y, card_w, card_h)
        
        tb = slide1.shapes.add_textbox(x + Inches(0.25), card_y + Inches(0.25), card_w - Inches(0.5), card_h - Inches(0.5))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        pm = tf.paragraphs[0]
        pm.text = metric
        pm.font.size = Pt(20)
        pm.font.bold = True
        pm.font.color.rgb = color
        pm.font.name = "Segoe UI"
        pm.space_after = Pt(6)

        pl = tf.add_paragraph()
        pl.text = label
        pl.font.size = Pt(12)
        pl.font.bold = True
        pl.font.color.rgb = TEXT_WHITE
        pl.font.name = "Segoe UI"
        pl.space_after = Pt(6)

        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_MUTED
        pd.font.name = "Segoe UI"

    slide1.notes_slide.notes_text_frame.text = (
        "SECTION 1: INTRODUCTION & PROJECT SCOPE\n\n"
        "Good morning, Doctor. Today I’m presenting my project: An End-to-End Handwritten Digit Recognition System and Interactive Real-Time Application.\n\n"
        "As we studied in Lecture 1, the central mission of Computer Vision is closing the semantic gap—bridging the distance between raw pixel values and high-level human semantic concepts.\n\n"
        "To explore this, I didn’t just train a single model. I built a rigorous comparative study across three distinct paradigms in Machine Learning history:\n"
        "1. A Classical Statistical Model: Support Vector Machine with an RBF kernel.\n"
        "2. A Fully-Connected Deep Neural Network (MLP).\n"
        "3. A Modern Convolutional Neural Network (CNN).\n\n"
        "I benchmarked all three on 21,000 test samples across accuracy, inference latency, parameter count, and memory footprint, and wrapped them into a production-grade Streamlit application deployed live on the cloud."
    )

    # =========================================================================
    # SLIDE 2: CV FOUNDATIONS & THE SEMANTIC GAP (LECTURES 1 & 2 EXTRA)
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "THEORETICAL FOUNDATION: LECTURE 1 & LECTURE 2 EXTRA", "Computer Vision Foundations & The Semantic Gap", CYAN)

    # Left Card: The Core Challenge
    c1 = add_card(slide2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.5), "The Semantic Gap Problem", CYAN)
    tb1 = slide2.shapes.add_textbox(Inches(1.05), Inches(2.1), Inches(5.1), Inches(3.6))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    bullets_s2_left = [
        ("Pixel Matrix vs. Perception:", "An image is simply a 2D matrix of integer brightness values [0-255]. A machine sees numbers; a human sees a '7' or a '3'."),
        ("The Traditional ML Bottleneck:", "Classical systems relied on handcrafted feature engineering (SIFT, HOG). If features are inadequate, the classifier fails regardless of capacity."),
        ("Failure of Naive Metrics:", "As proven in Lecture 2 Extra, naive distances (L1/L2 Euclidean norm) measure background pixel shifts rather than visual semantic identity.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets_s2_left):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        p.text = f"- {b_title} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.font.name = "Segoe UI"
        p.space_after = Pt(4)
        
        # Add description text
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT
        run.font.size = Pt(11)

    # Right Card: Visual Variation Challenges
    c2 = add_card(slide2, Inches(6.8), Inches(1.5), Inches(5.7), Inches(4.5), "Visual Variation Challenges (Johnson CS231n)", INDIGO)
    tb2 = slide2.shapes.add_textbox(Inches(7.05), Inches(2.1), Inches(5.2), Inches(3.6))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    
    bullets_s2_right = [
        ("Translation Variance:", "A user draws a digit in the top-left vs. center. Pixel-level coordinates change completely."),
        ("Scale & Aspect Ratio:", "Drawn digits vary wildly in height and width, requiring invariant bounding-box normalization."),
        ("Stroke Thickness & Style:", "Intra-class handwriting diversity: digit '1' can be a straight vertical line or have an angled serif."),
        ("Background & Clutter:", "Digital canvases introduce alpha transparency and border padding artifacts that must be stripped.")
    ]
    for i, (b_title, b_desc) in enumerate(bullets_s2_right):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = f"- {b_title} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.font.name = "Segoe UI"
        p.space_after = Pt(4)
        
        run = p.add_run()
        run.text = b_desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT
        run.font.size = Pt(11)

    # Bottom Callout Banner
    bot = add_card(slide2, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.85))
    tb_bot = slide2.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.65))
    tf_bot = tb_bot.text_frame
    tf_bot.word_wrap = True
    p_b = tf_bot.paragraphs[0]
    p_b.text = "CORE TAKEAWAY: "
    p_b.font.bold = True
    p_b.font.size = Pt(11)
    p_b.font.color.rgb = EMERALD
    p_b.font.name = "Segoe UI"
    run_b = p_b.add_run()
    run_b.text = "Closing the semantic gap requires robust data engineering BEFORE the model, paired with inductive bias (weight sharing & receptive fields) INSIDE the model."
    run_b.font.bold = False
    run_b.font.color.rgb = TEXT_WHITE
    run_b.font.size = Pt(11)

    slide2.notes_slide.notes_text_frame.text = (
        "SECTION 1 & 2 TRANSITION: THE SEMANTIC GAP & CV FOUNDATIONS\n\n"
        "Doctor, in Lecture 1 and Lecture 2 Extra, you highlighted the fundamental challenge of Computer Vision: the semantic gap.\n\n"
        "To a computer, an image is merely a 2D matrix of numbers between 0 and 255. When a human looks at an image, we instantly grasp the semantic concept of a digit, regardless of whether it's shifted, scaled, or written with a thick stroke.\n\n"
        "In the pre-deep-learning era, engineers tried to solve this with hand-crafted features or naive distance functions like L1 or L2 Euclidean distance. But as Lecture 2 Extra showed, Euclidean distance fails completely because two identical digits shifted by just two pixels will yield a massive pixel distance.\n\n"
        "To overcome this, we must handle visual challenges in two stages: first, through rigorous input preprocessing, and second, through architectural inductive bias in neural networks."
    )

    # =========================================================================
    # SLIDE 3: DATA PREPROCESSING & CENTER-OF-MASS (LECUN 1998 STANDARD)
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "DATA PIPELINE & PREPROCESSING: LECUN 1998 PROTOCOL", "Eliminating Translation Noise: The 5-Stage Pipeline", EMERALD)

    # Left Card: The Problem
    add_card(slide3, Inches(0.8), Inches(1.5), Inches(4.5), Inches(5.4), "The Canvas Distortion Problem", ROSE)
    tb_p3 = slide3.shapes.add_textbox(Inches(1.05), Inches(2.1), Inches(4.0), Inches(4.5))
    tf_p3 = tb_p3.text_frame
    tf_p3.word_wrap = True
    
    p_distort = [
        "Why raw drawings fail:",
        "When real users draw digits on an interactive HTML5 canvas, the drawings are arbitrarily placed (e.g. top-left corner) and have variable aspect ratios.",
        "The MNIST training distribution:",
        "Original MNIST images (LeCun 1998) are strictly pre-centered using the digit's center of mass into a 28x28 bounding box.",
        "The Consequence:",
        "If a drawing is off-center by even 4 pixels, fully-connected and convolutional layers experience severe distribution shift, resulting in complete classification failure."
    ]
    for i, line in enumerate(p_distort):
        p = tf_p3.paragraphs[0] if i == 0 else tf_p3.add_paragraph()
        if i % 2 == 0:
            p.text = line
            p.font.bold = True
            p.font.size = Pt(12)
            p.font.color.rgb = TEXT_WHITE
            p.space_after = Pt(2)
        else:
            p.text = line
            p.font.bold = False
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MUTED
            p.space_after = Pt(10)

    # Right Card: The 5-Step Pipeline (src/image_utils.py)
    add_card(slide3, Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.4), "The 5-Step Preprocessing Protocol (src/image_utils.py)", EMERALD)
    tb_steps = slide3.shapes.add_textbox(Inches(5.85), Inches(2.1), Inches(6.4), Inches(4.5))
    tf_steps = tb_steps.text_frame
    tf_steps.word_wrap = True

    steps_data = [
        ("Step 1: Grayscale & Inversion", "Strip RGBA alpha channel, convert to grayscale, and invert so the digit is white (255) on a black background (0)."),
        ("Step 2: Dynamic Bounding Box", "Locate non-zero rows and columns to crop out all empty canvas borders, isolating the active drawn digit."),
        ("Step 3: Aspect-Preserved 20x20 Fit", "Resize the cropped bounding box to fit inside a 20x20 square while strictly preserving original aspect ratio."),
        ("Step 4: Center-of-Mass Alignment", "Embed the 20x20 digit inside a 28x28 canvas. Using scipy.ndimage.center_of_mass, compute mass centroid (cy, cx) and translate the digit so its center of mass sits at exactly (14, 14)."),
        ("Step 5: Intensity Normalization", "Cast uint8 pixel values [0, 255] to float32 values in [0.0, 1.0], matching training tensor bounds.")
    ]
    for i, (title_st, desc_st) in enumerate(steps_data):
        p = tf_steps.paragraphs[0] if i == 0 else tf_steps.add_paragraph()
        p.text = f"{title_st}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = CYAN if i == 3 else TEXT_WHITE
        p.font.name = "Segoe UI"
        p.space_after = Pt(2)

        run = p.add_run()
        run.text = desc_st
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    slide3.notes_slide.notes_text_frame.text = (
        "SECTION 2: DATA ENGINEERING & THE CENTERING SECRET\n\n"
        "Doctor, before touching any model architecture, I focused heavily on Data Engineering, which Lectures 1 and 2 Extra define as the foundation of successful AI.\n\n"
        "When a user draws a digit on a digital canvas, it never matches the MNIST training distribution out-of-the-box. Users draw in corners, draw small or huge digits, or use thin strokes.\n\n"
        "To solve this translation variance without needing millions of augmented images, I implemented Yann LeCun’s exact 1998 preprocessing standard in `image_utils.py`:\n"
        "1. We strip alpha transparency and invert the colors.\n"
        "2. We extract the digit's bounding box and crop out empty borders.\n"
        "3. We resize the digit into a 20x20 box while preserving aspect ratio.\n"
        "4. Here is the key secret: we calculate the digit's center of mass using scipy.ndimage.center_of_mass, and translate it so its center of mass is locked precisely at (14, 14) in the 28x28 canvas.\n"
        "5. Finally, we normalize pixel intensities from [0, 255] to [0.0, 1.0].\n\n"
        "This mathematical step alone eliminates translation noise and dramatically boosts real-world interactive recognition accuracy."
    )

    # =========================================================================
    # SLIDE 4: MODEL 1 - CLASSICAL ML (RBF SVM)
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "PARADIGM 1: STATISTICAL MACHINE LEARNING", "Classical ML: Support Vector Machine (RBF Kernel)", AMBER)

    # Left Card: Mathematical Formulation
    add_card(slide4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.5), "Formulation & Optimization Theory", AMBER)
    tb_s4_l = slide4.shapes.add_textbox(Inches(1.05), Inches(2.1), Inches(5.1), Inches(3.6))
    tf_s4_l = tb_s4_l.text_frame
    tf_s4_l.word_wrap = True

    svm_theory = [
        ("Soft-Margin Objective:", "Maximizes the functional margin between classes while penalizing margin violations using slack variables xi_i: min 0.5||w||^2 + C * sum(xi_i)."),
        ("The RBF Kernel Trick:", "Raw 784D pixel space is not linearly separable. The Radial Basis Function (Gaussian) kernel K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2) projects data into an infinite-dimensional Hilbert space where classes become separable."),
        ("Hyperparameter Tuning:", "Tuned using 3-fold GridSearchCV over C in [0.1, 1.0, 10.0] and gamma in ['scale', 'auto']. Optimal parameters: C=10.0, gamma='scale'.")
    ]
    for i, (t_m, d_m) in enumerate(svm_theory):
        p = tf_s4_l.paragraphs[0] if i == 0 else tf_s4_l.add_paragraph()
        p.text = f"- {t_m} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = d_m
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    # Right Card: The Empirical Trade-Off
    add_card(slide4, Inches(6.8), Inches(1.5), Inches(5.7), Inches(4.5), "The Non-Parametric Bottleneck", ROSE)
    tb_s4_r = slide4.shapes.add_textbox(Inches(7.05), Inches(2.1), Inches(5.2), Inches(3.6))
    tf_s4_r = tb_s4_r.text_frame
    tf_s4_r.word_wrap = True

    svm_metrics = [
        ("Strong Test Accuracy: 97.02%", "Demonstrates that classical margin maximization is surprisingly effective on preprocessed digit pixels."),
        ("Storage Footprint: 21.3 MB (Huge)", "Because SVM is non-parametric, the trained model must store thousands of 784-dimensional support vector arrays. It cannot compress knowledge into small fixed weight matrices."),
        ("Inference Latency: 2.46 ms/sample (Slow)", "Over 15x to 20x slower than the neural networks. At prediction time, it must compute kernel distances against every stored support vector."),
        ("Edge Device Limitation:", "High memory overhead and compute latency make RBF SVM poorly suited for real-time edge devices.")
    ]
    for i, (t_m, d_m) in enumerate(svm_metrics):
        p = tf_s4_r.paragraphs[0] if i == 0 else tf_s4_r.add_paragraph()
        p.text = f"- {t_m}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = AMBER if i == 0 else (ROSE if i == 1 else TEXT_WHITE)
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = d_m
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    # Bottom Banner
    bot4 = add_card(slide4, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.85))
    tb_bot4 = slide4.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.65))
    tf_bot4 = tb_bot4.text_frame
    tf_bot4.word_wrap = True
    p_b4 = tf_bot4.paragraphs[0]
    p_b4.text = "TAKEAWAY: "
    p_b4.font.bold = True
    p_b4.font.size = Pt(11)
    p_b4.font.color.rgb = AMBER
    p_b4.font.name = "Segoe UI"
    run_b4 = p_b4.add_run()
    run_b4.text = "RBF SVM validates classical statistical theory, but its non-parametric nature causes severe memory (21.3 MB) and latency (2.46 ms) bottlenecks compared to deep learning."
    run_b4.font.bold = False
    run_b4.font.color.rgb = TEXT_WHITE
    run_b4.font.size = Pt(11)

    slide4.notes_slide.notes_text_frame.text = (
        "SECTION 3: MODEL 1 - CLASSICAL ML (RBF SVM)\n\n"
        "Doctor, our first benchmark represents the classical statistical era: Support Vector Machines (SVM).\n\n"
        "Because raw 784-dimensional pixel space is not linearly separable, a linear hyperplane underfits. Therefore, I applied the Radial Basis Function (RBF) kernel--the dual kernel trick from Lecture 2 Extra that maps the inputs into an infinite-dimensional feature space to maximize the separation margin.\n\n"
        "I tuned the hyperparameters using a 3-fold GridSearchCV, finding the optimal regularization parameter C=10.0 and gamma='scale'.\n\n"
        "The Results and Trade-offs:\n"
        "1. It achieved 97.02% accuracy, which is very respectable for classical ML.\n"
        "2. However, because SVM is a non-parametric model, it must store thousands of support vectors on disk. This blew up its model size to 21.3 Megabytes!\n"
        "3. Furthermore, its inference latency was 2.46 milliseconds per sample--over 15 to 20 times slower than the neural networks, because it has to compute kernel distances against all support vectors at test time."
    )

    # =========================================================================
    # SLIDE 5: MODEL 2 - DEEP MLP & DENSE BOTTLENECKS (LECTURES 2 & 3)
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "PARADIGM 2: DEEP FEEDFORWARD NEURAL NETWORKS", "Deep Multilayer Perceptron (MLP) & Dense Limitations", INDIGO)

    # Left Card: Topology & Parameter Explosion
    add_card(slide5, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.5), "Architecture & Parameter Explosion (Lecture 2)", INDIGO)
    tb_s5_l = slide5.shapes.add_textbox(Inches(1.05), Inches(2.1), Inches(5.1), Inches(3.6))
    tf_s5_l = tb_s5_l.text_frame
    tf_s5_l.word_wrap = True

    mlp_arch = [
        ("Layer Topology:", "Input (784) -> Dense(512) -> Dense(256) -> Dense(128) -> Dense(10, Softmax)."),
        ("Dense Parameter Explosion:", "Contains 571,530 trainable parameters. The first dense layer alone requires 784 x 512 + 512 = 401,920 weights (70% of the entire model!)."),
        ("Loss of Spatial Adjacency:", "Flattening the 2D grid (28x28) into a 1D vector (784,) permanently destroys pixel neighbor relationships. A pixel at (0, 27) is treated as distant from (1, 0) even though they are vertically adjacent in 2D space.")
    ]
    for i, (t_m, d_m) in enumerate(mlp_arch):
        p = tf_s5_l.paragraphs[0] if i == 0 else tf_s5_l.add_paragraph()
        p.text = f"- {t_m} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = d_m
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    # Right Card: Modern Regularization (Lecture 3)
    add_card(slide5, Inches(6.8), Inches(1.5), Inches(5.7), Inches(4.5), "Modern Stabilization Strategies (Lecture 3)", CYAN)
    tb_s5_r = slide5.shapes.add_textbox(Inches(7.05), Inches(2.1), Inches(5.2), Inches(3.6))
    tf_s5_r = tb_s5_r.text_frame
    tf_s5_r.word_wrap = True

    mlp_reg = [
        ("Batch Normalization (Ioffe & Szegedy 2015):", "Inserted after every hidden layer. Normalizes mini-batch activations to zero mean and unit variance, mitigating internal covariate shift and smoothing the loss surface."),
        ("Dropout Regularization (p = 0.4):", "Randomly zeroes 40% of hidden activations during training, forcing redundant feature representations and preventing neuron co-adaptation."),
        ("Peak Test Accuracy: 99.59%:", "Achieved highest benchmark accuracy on the 21,000-sample test set."),
        ("Ultra-Fast Latency: 0.12 ms/sample:", "Pure matrix multiplication allows near-instantaneous forward passes.")
    ]
    for i, (t_m, d_m) in enumerate(mlp_reg):
        p = tf_s5_r.paragraphs[0] if i == 0 else tf_s5_r.add_paragraph()
        p.text = f"- {t_m} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = EMERALD if i == 2 else TEXT_WHITE
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = d_m
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    # Bottom Banner
    bot5 = add_card(slide5, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.85))
    tb_bot5 = slide5.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.65))
    tf_bot5 = tb_bot5.text_frame
    tf_bot5.word_wrap = True
    p_b5 = tf_bot5.paragraphs[0]
    p_b5.text = "TAKEAWAY: "
    p_b5.font.bold = True
    p_b5.font.size = Pt(11)
    p_b5.font.color.rgb = INDIGO
    p_b5.font.name = "Segoe UI"
    run_b5 = p_b5.add_run()
    run_b5.text = "Deep MLP achieves top benchmark accuracy (99.59%) and lowest latency (0.12 ms), but suffers from parameter explosion (571k params) due to the lack of spatial inductive bias."
    run_b5.font.bold = False
    run_b5.font.color.rgb = TEXT_WHITE
    run_b5.font.size = Pt(11)

    slide5.notes_slide.notes_text_frame.text = (
        "SECTION 4: MODEL 2 — MULTILAYER PERCEPTRON (MLP) & DENSE LIMITATIONS\n\n"
        "Next, I constructed a deep Multilayer Perceptron with three hidden layers: 512, 256, and 128 units.\n\n"
        "In Lecture 2, you explained why fully-connected layers suffer from the parameter explosion: flattening a 28x28 grid into a 784-element vector discards all 2D spatial adjacency. Every neuron connects to every single pixel. Because of this, my MLP has over 571,000 parameters.\n\n"
        "To train such a deep network without vanishing gradients or internal covariate shift, I integrated two modern deep learning breakthroughs from Lecture 3:\n"
        "1. Batch Normalization after every linear layer to normalize activations, smooth the loss landscape, and accelerate convergence.\n"
        "2. Dropout (p=0.4) to break co-adaptation between neurons and prevent overfitting.\n\n"
        "The MLP reached 99.59% accuracy and is blazing fast at 0.12 milliseconds per sample, but it remains structurally vulnerable if digits are rotated or deformed."
    )

    # =========================================================================
    # SLIDE 6: MODEL 3 - CONVOLUTIONAL NEURAL NETWORK (LENET & VGG)
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "PARADIGM 3: SPATIAL DEEP LEARNING (CONVNETS)", "Convolutional Neural Network: Spatial Invariance & Efficiency", CYAN)

    # Left Card: Core Inductive Biases
    add_card(slide6, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.5), "Core Inductive Biases (Lecture 2)", CYAN)
    tb_s6_l = slide6.shapes.add_textbox(Inches(1.05), Inches(2.1), Inches(5.1), Inches(3.6))
    tf_s6_l = tb_s6_l.text_frame
    tf_s6_l.word_wrap = True

    cnn_biases = [
        ("Preserving 2D Topology:", "Input tensor maintains full spatial shape (28, 28, 1), preserving local adjacency and geometry."),
        ("Local Receptive Fields:", "Neurons connect only to small local patches (3x3), detecting primitive visual edges and strokes rather than full-image combinations."),
        ("Weight Sharing (Equivariance):", "The exact same 3x3 filter slides across the entire image. If a horizontal stroke appears in the top-left or bottom-right, the same filter activates."),
        ("Max Pooling Downsampling:", "2x2 pooling windows with stride 2 cut spatial dimensions in half, discarding 75% of activations while injecting local translation invariance.")
    ]
    for i, (t_m, d_m) in enumerate(cnn_biases):
        p = tf_s6_l.paragraphs[0] if i == 0 else tf_s6_l.add_paragraph()
        p.text = f"- {t_m} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = d_m
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    # Right Card: VGG Architecture & Parameter Proof
    add_card(slide6, Inches(6.8), Inches(1.5), Inches(5.7), Inches(4.5), "VGG Stacking & Parameter Efficiency (Lecture 3)", EMERALD)
    tb_s6_r = slide6.shapes.add_textbox(Inches(7.05), Inches(2.1), Inches(5.2), Inches(3.6))
    tf_s6_r = tb_s6_r.text_frame
    tf_s6_r.word_wrap = True

    cnn_vgg = [
        ("The VGG 3x3 Principle:", "Instead of using large 5x5 filters (LeNet-5), I stacked two consecutive 3x3 convolutions in each block (32 and 64 filters)."),
        ("Mathematical Equivalence:", "Two stacked 3x3 filters have an effective receptive field of 1 + 2 + 2 = 5x5, but with TWO non-linear ReLUs instead of one!"),
        ("28% Weight Reduction:", "Two 3x3 filters use 2 x (3^2) = 18 weights vs. 5^2 = 25 weights for a single 5x5 filter (28% fewer parameters)."),
        ("Empirical Efficiency:", "Total parameters: 421,642 (26% FEWER than the MLP's 571k params!). Test accuracy: 99.09%, size: 4.86 MB.")
    ]
    for i, (t_m, d_m) in enumerate(cnn_vgg):
        p = tf_s6_r.paragraphs[0] if i == 0 else tf_s6_r.add_paragraph()
        p.text = f"- {t_m} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = EMERALD if i == 3 else TEXT_WHITE
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = d_m
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    # Bottom Banner
    bot6 = add_card(slide6, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.85))
    tb_bot6 = slide6.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.65))
    tf_bot6 = tb_bot6.text_frame
    tf_bot6.word_wrap = True
    p_b6 = tf_bot6.paragraphs[0]
    p_b6.text = "TAKEAWAY: "
    p_b6.font.bold = True
    p_b6.font.size = Pt(11)
    p_b6.font.color.rgb = CYAN
    p_b6.font.name = "Segoe UI"
    run_b6 = p_b6.add_run()
    run_b6.text = "The CNN demonstrates the power of spatial inductive bias: matching MLP accuracy (99.09% vs 99.59%) while cutting parameter count by 26% (421k vs 571k) and model size by 25%."
    run_b6.font.bold = False
    run_b6.font.color.rgb = TEXT_WHITE
    run_b6.font.size = Pt(11)

    slide6.notes_slide.notes_text_frame.text = (
        "SECTION 5: MODEL 3 — CONVOLUTIONAL NEURAL NETWORK (CNN)\n\n"
        "To fundamentally solve the spatial limitation, I designed our Convolutional Neural Network.\n\n"
        "Architecturally, it is a modernized evolution of LeNet-5 (1998) and incorporates the core design philosophy of VGGNet (Simonyan & Zisserman 2014) from Lecture 3:\n"
        "Instead of large 5x5 or 7x7 filters, I stacked two consecutive 3x3 convolutions in each block. As we learned in Lecture 3, two stacked 3x3 layers have the exact same effective receptive field as a single 5x5 layer, but with two non-linear ReLU activations instead of one, and 18 weights versus 25 weights—a 28% parameter reduction!\n\n"
        "The CNN utilizes the two fundamental inductive biases from Lecture 2:\n"
        "1. Local Receptive Fields to extract localized strokes and edges.\n"
        "2. Weight Sharing so the same filter scans the whole image, conferring translation equivariance.\n\n"
        "The Critical Comparison:\n"
        "The CNN achieved 99.09% accuracy, but it did so with only 421,000 parameters—26% fewer parameters than the MLP, and a 25% smaller file size (4.8 MB vs 6.5 MB), completely validating the parameter efficiency of convolutions taught in our lectures."
    )

    # =========================================================================
    # SLIDE 7: EMPIRICAL BENCHMARKS & CROSS-MODEL COMPARISON TABLE
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "CROSS-MODEL BENCHMARK MATRIX (21,000 TEST SAMPLES)", "Empirical Performance & Architectural Trade-Offs", EMERALD)

    # Comparison Table Shape
    table_shape = slide7.shapes.add_table(4, 7, Inches(0.8), Inches(1.5), Inches(11.73), Inches(2.2))
    table = table_shape.table

    # Set column widths
    col_widths = [Inches(1.8), Inches(1.8), Inches(1.6), Inches(1.8), Inches(1.6), Inches(1.5), Inches(1.63)]
    for idx, width in enumerate(col_widths):
        table.columns[idx].width = width

    headers = ["Model", "Paradigm", "Accuracy", "Latency / Sample", "Parameters", "File Size", "Spatial Bias"]
    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.alignment = PP_ALIGN.CENTER
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = CYAN
        p.font.name = "Segoe UI"

    rows_data = [
        ["Classical SVM", "Statistical (RBF)", "97.02%", "2.46 ms", "Non-parametric", "21.3 MB", "None (1D vector)"],
        ["Deep MLP", "Dense-Only", "99.59%", "0.12 ms", "571,530", "6.57 MB", "None (Flattened)"],
        ["Modern CNN", "Convolutional", "99.09%", "0.16 ms", "421,642", "4.86 MB", "High (2D Receptive)"]
    ]
    for r_idx, row in enumerate(rows_data, 1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(18, 24, 38) if r_idx % 2 == 1 else RGBColor(24, 32, 47)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.alignment = PP_ALIGN.CENTER
            p.font.size = Pt(11)
            p.font.name = "Segoe UI"
            if c_idx == 0:
                p.font.bold = True
                p.font.color.rgb = TEXT_WHITE
            elif c_idx == 2:
                p.font.bold = True
                p.font.color.rgb = EMERALD if "99" in val else AMBER
            elif c_idx == 4:
                p.font.color.rgb = CYAN if "421" in val else TEXT_LIGHT
            else:
                p.font.color.rgb = TEXT_LIGHT

    # 3 Summary Cards below table
    cards_summary = [
        ("Accuracy vs Generalization", "Deep MLP and CNN both surpass 99% accuracy. While MLP scored slightly higher on preprocessed test digits, CNN exhibits vastly superior resilience to off-center and rotated inputs.", INDIGO),
        ("Speed vs Memory Trade-Off", "Neural networks perform constant-time O(1) matrix operations, running in 0.12-0.16 ms. SVM must compute kernel dot products against thousands of support vectors, costing 2.46 ms.", AMBER),
        ("Parameter Efficiency Champion", "The CNN achieves near-identical accuracy to the MLP while utilizing 26% fewer parameters (421k vs 571k) and a 25% smaller disk footprint (4.8 MB vs 6.5 MB).", CYAN)
    ]
    card_w7 = Inches(3.7)
    card_h7 = Inches(2.7)
    card_y7 = Inches(4.1)

    for i, (m_title, m_desc, m_color) in enumerate(cards_summary):
        x = Inches(0.8) + i * Inches(4.0)
        add_card(slide7, x, card_y7, card_w7, card_h7, m_title, m_color)
        tb = slide7.shapes.add_textbox(x + Inches(0.25), card_y7 + Inches(0.7), card_w7 - Inches(0.5), card_h7 - Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = m_desc
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_LIGHT
        p.font.name = "Segoe UI"

    slide7.notes_slide.notes_text_frame.text = (
        "SECTION 6: EMPIRICAL BENCHMARKS & CROSS-MODEL COMPARISONS\n\n"
        "Doctor, this benchmark table summarizes our unified empirical evaluation on 21,000 real test samples:\n\n"
        "1. Classical SVM proved that margin maximization works (97.02%), but its non-parametric nature requires storing thousands of support vectors—leading to a massive 21.3 MB file and slow 2.46 ms latency.\n"
        "2. Deep MLP achieves top accuracy (99.59%) and lowest latency (0.12 ms), but is burdened with 571k parameters because dense layers discard 2D geometry.\n"
        "3. Modern CNN provides the optimal structural balance: matching accuracy (99.09%) with 26% fewer parameters (421k) and 25% smaller file size (4.86 MB).\n\n"
        "This clearly proves that architectural inductive bias is superior to brute-force dense parameter stacking."
    )

    # =========================================================================
    # SLIDE 8: INTERACTIVE WEB APP & PRODUCTION CLOUD DEPLOYMENT
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "DELIVERABLE: PRODUCTION STREAMLIT WEB APP", "Interactive Cloud Deployment & Real-Time UX", INDIGO)

    # Left Card: Cloud Architecture
    add_card(slide8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(4.5), "Production Architecture (app.py)", INDIGO)
    tb_s8_l = slide8.shapes.add_textbox(Inches(1.05), Inches(2.1), Inches(5.1), Inches(3.6))
    tf_s8_l = tb_s8_l.text_frame
    tf_s8_l.word_wrap = True

    app_features = [
        ("Streamlit Community Cloud:", "Publicly hosted and live at: https://mnistdigitrecognizerr.streamlit.app/ with automated GitHub CI/CD syncing."),
        ("Interactive HTML5 Canvas:", "Embedded drawing pad with dynamic stroke width control, dark canvas theme, and instant 'Clear' / 'Predict' triggers."),
        ("Simultaneous 3-Model Inference:", "A single click on 'Predict' queries CNN, MLP, and Classical SVM concurrently, executing in less than 50 milliseconds end-to-end."),
        ("Defensive Preprocessing Cache:", "Optimized image tensor pipeline prevents redundant canvas redraws and maintains smooth user experience.")
    ]
    for i, (t_m, d_m) in enumerate(app_features):
        p = tf_s8_l.paragraphs[0] if i == 0 else tf_s8_l.add_paragraph()
        p.text = f"- {t_m} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = d_m
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    # Right Card: UI/UX & Interpretability
    add_card(slide8, Inches(6.8), Inches(1.5), Inches(5.7), Inches(4.5), "Interpretability & Model Consensus", CYAN)
    tb_s8_r = slide8.shapes.add_textbox(Inches(7.05), Inches(2.1), Inches(5.2), Inches(3.6))
    tf_s8_r = tb_s8_r.text_frame
    tf_s8_r.word_wrap = True

    app_ux = [
        ("Visual Preprocessing Preview:", "The sidebar displays the exact 28x28 normalized tensor, showing the user how the center-of-mass shift centered their drawn stroke."),
        ("Ensemble Consensus Badge:", "An automated agreement indicator highlights whether all 3 models agree ('Full Consensus (3/3)') or if a model diverges ('Split Decision')."),
        ("Candidate Probability Bars:", "Visual probability distributions display Top-3 prediction confidence for each model, exposing borderline ambiguities (e.g. 4 vs 9)."),
        ("Modern Glassmorphic Dark UI:", "Custom CSS styling with cyber cyan accents and card containers designed for academic presentations.")
    ]
    for i, (t_m, d_m) in enumerate(app_ux):
        p = tf_s8_r.paragraphs[0] if i == 0 else tf_s8_r.add_paragraph()
        p.text = f"- {t_m} "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = EMERALD if i == 1 else TEXT_WHITE
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = d_m
        run.font.bold = False
        run.font.size = Pt(11)
        run.font.color.rgb = TEXT_LIGHT

    # Bottom Banner
    bot8 = add_card(slide8, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.85))
    tb_bot8 = slide8.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.65))
    tf_bot8 = tb_bot8.text_frame
    tf_bot8.word_wrap = True
    p_b8 = tf_bot8.paragraphs[0]
    p_b8.text = "LIVE DEPLOYMENT: "
    p_b8.font.bold = True
    p_b8.font.size = Pt(11)
    p_b8.font.color.rgb = CYAN
    p_b8.font.name = "Segoe UI"
    run_b8 = p_b8.add_run()
    run_b8.text = "The live web application is fully operational on Streamlit Community Cloud at https://mnistdigitrecognizerr.streamlit.app/ — open for interactive audience testing."
    run_b8.font.bold = False
    run_b8.font.color.rgb = TEXT_WHITE
    run_b8.font.size = Pt(11)

    slide8.notes_slide.notes_text_frame.text = (
        "SECTION 7: LIVE APP DEMO & PRODUCTION DEPLOYMENT\n\n"
        "Finally, Doctor, I deployed the entire project to production as an interactive cloud web application using Streamlit Community Cloud.\n\n"
        "Users can draw any digit on the interactive canvas. In real time, the application:\n"
        "1. Shows the side-by-side center-of-mass preprocessed 28x28 tensor.\n"
        "2. Executes simultaneous inference across CNN, MLP, and SVM.\n"
        "3. Displays confidence bar charts and an Ensemble Consensus Badge indicating whether all three models agree.\n\n"
        "You can test it directly on your phone or laptop at: https://mnistdigitrecognizerr.streamlit.app/."
    )

    # =========================================================================
    # SLIDE 9: ORAL DEFENSE CHEAT-SHEET: TECHNICAL QUESTIONS (PART 1)
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "EXAMINATION DEFENSE: FREQUENT TECHNICAL QUESTIONS", "Oral Defense Cheat-Sheet: Core Technical Q&A", ROSE)

    qa_list_1 = [
        ("Q1: Why is SVM 21.3 MB while CNN is only 4.8 MB?",
         [
             ("Core Cause:", "SVM is non-parametric; it does not store fixed weight matrices, but rather serializes thousands of 784-float training support vectors."),
             ("Empirical Footprint:", "Over 5,000 support vectors x 784 floats x 4 bytes = 21.3 MB on disk."),
             ("Contrast with CNN:", "CNN stores only 421k fixed weight scalars (4.8 MB), independent of training set size."),
             ("Lecture Context:", "Lecture 1 & Lecture 2 Extra (Statistical Learning & Non-Parametric Limits).")
         ],
         AMBER),
        ("Q2: Why stack two 3x3 filters instead of one 5x5? (VGG)",
         [
             ("Mathematical Proof:", "Effective receptive field: 1 + 2 + 2 = 5x5, matching a 5x5 filter identically."),
             ("Parameter Savings:", "Two 3x3 filters require 2 x (3^2 x C^2) = 18 C^2 weights vs. 5^2 C^2 = 25 C^2 weights (-28% parameter reduction!)."),
             ("Non-Linear Expressiveness:", "Two ReLUs instead of one allows the network to learn richer, more discriminative feature representations."),
             ("Lecture Context:", "Lecture 3 (Simonyan & Zisserman 2014, VGGNet Architecture).")
         ],
         CYAN),
        ("Q3: Why did user drawings fail before Centering?",
         [
             ("The Semantic Gap:", "Canvas drawings have arbitrary positions, violating MNIST's strict centroid distribution."),
             ("Euclidean Distance Fragility:", "Lecture 2 Extra proved that pixel-space shifts cause huge L1/L2 distance errors."),
             ("LeCun 1998 Protocol:", "Bounding box crop + 20x20 resize + center-of-mass shift to (14, 14) eliminates translation noise."),
             ("Interactive Result:", "Drawings anywhere on the canvas are mapped to the exact training distribution.")
         ],
         EMERALD)
    ]
    card_w9 = Inches(3.7)
    card_h9 = Inches(4.5)
    card_y9 = Inches(1.5)

    for i, (q_text, bullets, q_color) in enumerate(qa_list_1):
        x = Inches(0.8) + i * Inches(4.0)
        add_card(slide9, x, card_y9, card_w9, card_h9, q_text, q_color)
        tb = slide9.shapes.add_textbox(x + Inches(0.2), card_y9 + Inches(0.75), card_w9 - Inches(0.4), card_h9 - Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        
        for b_idx, (b_hdr, b_txt) in enumerate(bullets):
            p = tf.paragraphs[0] if b_idx == 0 else tf.add_paragraph()
            p.text = f"- {b_hdr} "
            p.font.bold = True
            p.font.size = Pt(10.5)
            p.font.color.rgb = TEXT_WHITE
            p.font.name = "Segoe UI"
            p.space_after = Pt(3)

            run = p.add_run()
            run.text = b_txt
            run.font.bold = False
            run.font.size = Pt(10)
            run.font.color.rgb = TEXT_LIGHT

    # Bottom Banner
    bot9 = add_card(slide9, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.85))
    tb_bot9 = slide9.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.65))
    tf_bot9 = tb_bot9.text_frame
    tf_bot9.word_wrap = True
    p_b9 = tf_bot9.paragraphs[0]
    p_b9.text = "KEY DEFENSE ADVICE: "
    p_b9.font.bold = True
    p_b9.font.size = Pt(11)
    p_b9.font.color.rgb = ROSE
    p_b9.font.name = "Segoe UI"
    run_b9 = p_b9.add_run()
    run_b9.text = "When answering the Doctor, cite the specific lecture authors (LeCun 1998, Simonyan & Zisserman 2014) to demonstrate deep mastery of the course syllabus."
    run_b9.font.bold = False
    run_b9.font.color.rgb = TEXT_WHITE
    run_b9.font.size = Pt(11)

    slide9.notes_slide.notes_text_frame.text = (
        "SECTION 8: ORAL DEFENSE CHEAT-SHEET (PART 1)\n\n"
        "Doctor's Question 1: 'Why does SVM have a 21 MB file size while CNN is only 4.8 MB?'\n"
        "Your Answer: 'Because SVM is non-parametric; it doesn't just store weights, it has to serialize thousands of 784-dimensional floating-point support vectors. The CNN only stores fixed weight matrices, which total only 421k floats.'\n\n"
        "Doctor's Question 2: 'Why stack two 3x3 filters instead of one 5x5?'\n"
        "Your Answer: 'From VGGNet in Lecture 3: Two 3x3 convolutions give the same 5x5 receptive field, but introduce two non-linearities (ReLUs) instead of one, and require only 18 weights compared to 25 weights--saving 28% parameters.'\n\n"
        "Doctor's Question 3: 'Why did your hand-drawn digits initially fail before preprocessing?'\n"
        "Your Answer: 'Because of the Semantic Gap and Translation Variance from Lecture 2 Extra. By implementing LeCun's 1998 protocol--cropping to a bounding box, aspect-ratio scaling to 20x20, and center-of-mass translation to (14, 14)--we eliminate spatial shift before classification.'"
    )

    # =========================================================================
    # SLIDE 10: ORAL DEFENSE CHEAT-SHEET: ARCHITECTURAL THEORY (PART 2)
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "EXAMINATION DEFENSE: ARCHITECTURAL THEORY & BEYOND", "Oral Defense Cheat-Sheet: Advanced Deep Learning Theory", CYAN)

    qa_list_2 = [
        ("Q4: Role of Batch Normalization? (Lecture 3)",
         [
             ("Mathematical Operation:", "Normalizes activations to zero mean & unit variance: x_hat = (x - mu_B) / sqrt(sigma_B^2 + eps)."),
             ("Learnable Parameters:", "Applies scale gamma and shift beta: y = gamma * x_hat + beta to recover optimal activation expressiveness."),
             ("Internal Covariate Shift:", "Stabilizes activation distribution across deep layers, preventing vanishing/exploding gradients."),
             ("Optimization Impact:", "Allows higher learning rates and acts as mild regularizer.")
         ],
         CYAN),
        ("Q5: What does the 'C' hyperparameter regulate?",
         [
             ("Objective Trade-Off:", "Controls balance between maximizing margin width and penalizing margin violations: min 0.5||w||^2 + C * sum(xi_i)."),
             ("High C (C = 10.0):", "Heavy penalty for misclassification. Creates a narrower, harder margin, fitting complex boundaries."),
             ("Low C (e.g. C = 0.1):", "Tolerates more margin errors in exchange for wider separation margin (softer margin)."),
             ("Our Selection:", "Cross-validation selected C=10.0 with RBF kernel to ensure high accuracy (97.02%).")
         ],
         AMBER),
        ("Q6: How to scale to deeper architectures?",
         [
             ("The Degradation Barrier:", "Beyond 20 layers, accuracy saturates and degrades rapidly due to vanishing gradients."),
             ("ResNet (He et al. 2015):", "Lecture 3: Skip connections y = F(x) + x enable identity mapping and direct gradient backpropagation."),
             ("GoogLeNet / Inception:", "Uses 1x1 convolutions for cross-channel dimensionality reduction before expensive 3x3/5x5 convolutions."),
             ("Modern Practice:", "Our CNN applies these core design rules at a scale suited for MNIST.")
         ],
         INDIGO)
    ]

    for i, (q_text, bullets, q_color) in enumerate(qa_list_2):
        x = Inches(0.8) + i * Inches(4.0)
        add_card(slide10, x, card_y9, card_w9, card_h9, q_text, q_color)
        tb = slide10.shapes.add_textbox(x + Inches(0.2), card_y9 + Inches(0.75), card_w9 - Inches(0.4), card_h9 - Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        
        for b_idx, (b_hdr, b_txt) in enumerate(bullets):
            p = tf.paragraphs[0] if b_idx == 0 else tf.add_paragraph()
            p.text = f"- {b_hdr} "
            p.font.bold = True
            p.font.size = Pt(10.5)
            p.font.color.rgb = TEXT_WHITE
            p.font.name = "Segoe UI"
            p.space_after = Pt(3)

            run = p.add_run()
            run.text = b_txt
            run.font.bold = False
            run.font.size = Pt(10)
            run.font.color.rgb = TEXT_LIGHT

    # Bottom Banner
    bot10 = add_card(slide10, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.85))
    tb_bot10 = slide10.shapes.add_textbox(Inches(1.0), Inches(6.3), Inches(11.3), Inches(0.65))
    tf_bot10 = tb_bot10.text_frame
    tf_bot10.word_wrap = True
    p_b10 = tf_bot10.paragraphs[0]
    p_b10.text = "CONCLUSION: "
    p_b10.font.bold = True
    p_b10.font.size = Pt(11)
    p_b10.font.color.rgb = CYAN
    p_b10.font.name = "Segoe UI"
    run_b10 = p_b10.add_run()
    run_b10.text = "Thank you, Doctor. The project successfully demonstrates the evolution from statistical machine learning to modern deep convolutional architectures. I welcome your questions."
    run_b10.font.bold = False
    run_b10.font.color.rgb = TEXT_WHITE
    run_b10.font.size = Pt(11)

    slide10.notes_slide.notes_text_frame.text = (
        "SECTION 8: ORAL DEFENSE CHEAT-SHEET (PART 2) & CLOSING\n\n"
        "Doctor's Question 4: 'What is the role of Batch Normalization?'\n"
        "Your Answer: 'From Ioffe & Szegedy 2015 (Lecture 3): Batch Normalization stabilizes activations across mini-batches, mitigates internal covariate shift, prevents vanishing/exploding gradients, and allows higher learning rates.'\n\n"
        "Doctor's Question 5: 'What does the C parameter in SVM control?'\n"
        "Your Answer: 'It is the penalty parameter for misclassification. A high C (C=10.0) prioritizes a narrow margin with minimal training errors, while a low C allows more margin violations for better generalization. I selected C=10 via 3-fold cross-validation.'\n\n"
        "Doctor's Question 6: 'How would you scale this to deeper networks?'\n"
        "Your Answer: 'As covered in Lecture 3, deeper networks suffer from vanishing gradients and degradation. ResNet solves this via residual skip connections (y = F(x) + x) allowing direct gradient flow, and Inception uses 1x1 convolutions for dimensionality reduction.'\n\n"
        "Closing Words: 'Thank you, Doctor. I am ready to answer any questions or demonstrate the live app.'"
    )

    output_path = "MNIST_Digit_Recognizer_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation successfully created and saved to: {output_path}")

if __name__ == "__main__":
    create_deck()

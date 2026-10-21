# Rebuilds the two-page models PDF. Uses Carlito and DejaVu fonts (Linux paths below); change F to your font folder on Windows/Mac.
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

F = "/usr/share/fonts/truetype/"
pdfmetrics.registerFont(TTFont("Carlito", F + "crosextra/Carlito-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Carlito-Bold", F + "crosextra/Carlito-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Carlito-Italic", F + "crosextra/Carlito-Italic.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu", F + "dejavu/DejaVuSans.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Carlito", normal="Carlito", bold="Carlito-Bold", italic="Carlito-Italic", boldItalic="Carlito-Bold")

PLUM = colors.HexColor("#4B1248")
PINK = colors.HexColor("#B5145D")
INK = colors.HexColor("#2A0B2F")
GREY = colors.HexColor("#67516B")
LILAC = colors.HexColor("#F5EEF7")
LINE = colors.HexColor("#E2D5E6")
HEAD = colors.HexColor("#EBDDF0")

STAR = '<font name="DejaVu" size="7.5">\u2605</font>'

body = ParagraphStyle("body", fontName="Carlito", fontSize=9.3, leading=11.55, textColor=INK, spaceAfter=3)
small = ParagraphStyle("small", parent=body, fontSize=8.4, leading=10.4, textColor=GREY)
cell = ParagraphStyle("cell", parent=body, fontSize=8.6, leading=10.4, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Carlito-Bold")
h1 = ParagraphStyle("h1", fontName="Carlito-Bold", fontSize=19, leading=22, textColor=PLUM, spaceAfter=2)
sub = ParagraphStyle("sub", parent=body, fontSize=9.6, textColor=GREY, spaceAfter=8)
h2 = ParagraphStyle("h2", fontName="Carlito-Bold", fontSize=12.5, leading=15, textColor=PLUM, spaceBefore=6, spaceAfter=3)
label = ParagraphStyle("label", parent=body, fontName="Carlito-Bold", textColor=PINK, spaceAfter=1)
flow = ParagraphStyle("flow", parent=body, fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=0)
arrow = ParagraphStyle("arrow", parent=body, fontName="DejaVu", fontSize=12, alignment=TA_CENTER, textColor=PINK, spaceAfter=0)


def P(t, s=body):
    return Paragraph(t, s)


def table(rows, widths, head=True, zebra=True):
    data = [[P(c, cellb if (head and i == 0) else cell) if isinstance(c, str) else c for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(st))
    return t


def box(flowables, width):
    t = Table([[flowables]], colWidths=[width], hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LILAC),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Carlito", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(16 * mm, 8 * mm, "Price Confidence, Team Edge Case (IIT Roorkee), Meesho DICE Challenge S3. Demo models trained on synthetic data.")
    canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, "Page %d" % doc.page)
    canvas.restoreState()


W = A4[0] - 32 * mm
doc = SimpleDocTemplate("Price_Confidence_Models_Explained.pdf", pagesize=A4,
                        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=10 * mm, bottomMargin=13 * mm,
                        title="Price Confidence: how the three models work", author="Team Edge Case, IIT Roorkee")
s = []

s.append(P("Price Confidence: how the three models work", h1))
s.append(P("What each model learns, how it is trained, what it outputs, and how it changes the price a seller sees. "
           "Numbers are from our demo, trained on 300 synthetic listings (150 soap, 150 t-shirt).", sub))

fw = (W - 3 * 7 * mm) / 4
chain = Table([[P("<b>A2 Visibility</b><br/>How many people<br/>see the listing", flow), P("\u2192", arrow),
                P("<b>A1 Conversion</b><br/>How many of them<br/>place an order", flow), P("\u2192", arrow),
                P("<b>B Keep share</b><br/>How many orders get paid,<br/>and how many leave a review", flow), P("\u2192", arrow),
                P("<b>Seller's own costs</b><br/>Profit for<br/>that week", flow)]],
              colWidths=[fw, 7 * mm, fw, 7 * mm, fw, 7 * mm, fw])
chain.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("BACKGROUND", (0, 0), (0, 0), LILAC), ("BACKGROUND", (2, 0), (2, 0), LILAC),
                           ("BACKGROUND", (4, 0), (4, 0), LILAC), ("BACKGROUND", (6, 0), (6, 0), HEAD),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
s.append(chain)
s.append(Spacer(1, 3))
s.append(P("New sales and reviews from this week feed next week's A2 and A1. The app repeats the chain week by week, for every candidate "
           "price, over 12 weeks (26 for the launch-lower test), for a good, a typical and a weak product.", small))

s.append(P("The training data and the one rule all three models follow", h2))
s.append(P("One row = one listing in one week: <b>3,909 soap rows and 3,718 t-shirt rows</b>, 150 listings each, launched Sep 2025 to "
           "Aug 2026 and followed to 21 Sep 2026. Inputs: <b>relative price</b> = price \u00f7 market median \u2212 1; <b>rating shown</b> that week, "
           "pulled toward the typical 3.9" + STAR + " when reviews are few; <b>reviews so far</b>; <b>recent sales</b> (conversion versus the category "
           "norm); launch phase; category demand; listing quality. <b>The rule:</b> every model gives each listing its own baseline "
           "(fixed effects), so effects are learnt from a listing changing over time, never from comparing good products with bad ones. "
           "A second step then puts those baselines on listing quality, so a brand-new listing gets a starting level.", body))

s.append(P("A1 Conversion: of the people who see it, how many order?", h2))
s.append(P("<b>Trained on:</b> orders in the week with impressions as the exposure (Poisson regression with a log-impressions offset), "
           "so more views are never mistaken for better conversion. <b>Why baselines matter:</b> comparing different listings would say "
           "+10% price costs only 7% of orders (soap): better products charge more and still sell more.", body))
s.append(table([
    ["What it learnt", "Soap", "T-shirt", "Meaning"],
    ["Price vs market", "\u22122.40 (curve +1.36)", "\u22123.20 (curve +1.39)", "<b>+10% price: \u221220% orders per impression</b> (soap), \u221226% (t-shirt)"],
    ["Rating shown, per star", "0.49", "0.65", "+0.3" + STAR + ": +16% conversion (soap), +21% (t-shirt)"],
    ["Reviews so far", "0.01", "0.00", "About zero: reviews work through visibility (A2)"],
    ["Listing quality, per SD", "0.68", "0.71", "Better-presented listings convert more"],
    ["Spread between products", "0.44", "0.43", "A top-quarter product converts about 35% better than a typical one"],
], [35 * mm, 29 * mm, 29 * mm, W - 93 * mm]))
s.append(Spacer(1, 2))
s.append(P("<b>How it moves the answer:</b> the demand curve. For Seller 6's soap in launch week, conversion is 0.125% at ₹179, 0.087% at "
           "₹209 and 0.070% at ₹229.", body))

s.append(P("A2 Visibility: how many people see the listing each week?", h2))
s.append(P("<b>Trained on:</b> weekly impressions, Poisson regression with listing baselines. The earlier version without baselines "
           "said recent sales barely affect visibility (0.007); within listings the effect is 0.30. Same trap as in A1.", body))
s.append(table([
    ["What it learnt", "Soap", "T-shirt", "Meaning"],
    ["Launch weeks 1\u20132 / 3\u20134", "0.59 / 0.32", "0.61 / 0.34", "A new listing is shown <b>1.8\u00d7</b>, then 1.4\u00d7, then normal"],
    ["Recent sales vs category norm", "0.30", "0.29", "<b>Selling at twice the norm: +35% impressions</b> next week"],
    ["Reviews so far", "0.10", "0.07", "Reviews double: +7% (soap), +5% (t-shirt)"],
    ["Rating shown, per star", "0.22", "0.54", "+0.3" + STAR + ": +7% (soap), +17% (t-shirt)"],
    ["Category demand / sale week", "1.07 / 0.25", "0.97 / 0.27", "Demand +10%: about +10%; sale week: +29% (soap)"],
], [35 * mm, 29 * mm, 29 * mm, W - 93 * mm]))
s.append(Spacer(1, 2))
s.append(P("<b>How it moves the answer:</b> Seller 6 is shown about 10,200 times in week 1 and about 5,400 by week 5. For a good product, "
           "launching at ₹129 brings about 4.6\u00d7 the orders in week 2, which lifts week-5 impressions by about 56%. That is real, but the discount costs more "
           "than the extra visibility earns back.", body))

s.append(KeepTogether([P("B Keep share: of the orders, how many get paid, and how many review?", h2), table([
    ["Part", "How it is trained", "Soap", "T-shirt"],
    ["RTO rate", "Total RTO \u00f7 total orders (it did not depend on price)", "10.1%", "12.5%"],
    ["Return rate", "Logistic regression with listing baselines; rises with price above the market; better listings return less",
     "6.3% at ₹209", "21% at ₹349"],
    ["Good vs weak product", "Spread of return rates between products, noise removed", "4.8% / 8.3%", "spread 0.38"],
    ["Returns resold / review rate", "Plain shares from the data", "0% / 14%", "70% / 15%"],
], [33 * mm, W - 33 * mm - 52 * mm, 26 * mm, 26 * mm])]))
s.append(Spacer(1, 2))
s.append(P("We force the price effect on returns to be zero or positive: left free, the model learnt that pricier listings get fewer "
           "returns, the same quality trap. <b>How it moves the answer:</b> (1 \u2212 10.1%) \u00d7 (1 \u2212 6%) = <b>84.5%</b> of Seller 6's "
           "parcels get paid; costs are divided by that, which gives his ₹145 floor. The review rate turns orders into next week's reviews.", body))

s.append(P("All three together: one simulated week", h2))
s.append(table([
    ["Step", "Model", "Seller 6, typical product, ₹209, week 1"],
    ["1. How many see it", "A2", "About 10,200 impressions (launch boost, no reviews yet)"],
    ["2. How many order", "A1", "0.087% conversion: about 8.9 orders placed"],
    ["3. How many pay", "B", "84.5% paid: about 7.5 paid orders; about 1.1 new reviews"],
    ["4. Profit", "Seller's costs", "₹51 per dispatched order \u00d7 8.9 = about ₹455 for the week"],
    ["5. Next week", "Loop", "Sales and reviews feed A2 and A1; the launch boost fades; repeat"],
], [36 * mm, 24 * mm, W - 60 * mm]))
s.append(Spacer(1, 4))
s.append(table([
    ["Launch price", "Orders placed (paid), 12 weeks", "Profit, 12 weeks", "Why"],
    ["₹179", "107 (90)", "₹2,973", "More orders, but each earns too little"],
    ["<b>₹209 (chosen)</b>", "<b>69 (58)</b>", "<b>₹3,537</b>", "<b>Within 5% of the best, with more orders, so reviews come sooner. 58 paid \u00d7 ₹61 = ₹3,537</b>"],
    ["₹229", "54 (46)", "₹3,627", "Highest profit, but fewer orders and slower reviews"],
], [26 * mm, 42 * mm, 28 * mm, W - 96 * mm]))

s.append(P("Checked against the simulator that generated the data", h2))
s.append(P("We ran Seller 6's soap through the simulator's own buyer rules, 300 times per plan, and compared the change in 26-week "
           "profit against simply starting at ₹209.", small))
s.append(table([
    ["Plan", "Good product: model / simulator", "Typical: model / simulator", "Weak: model / simulator"],
    ["₹129 for 4 weeks (below the ₹145 floor)", "\u2212₹2,626 / \u2212₹779", "\u2212₹2,249 / \u2212₹1,308", "\u2212₹1,688 / \u2212₹1,698"],
    ["₹169 for 4 weeks (above the floor)", "+₹72 / +₹1,690", "\u2212₹241 / +₹531", "\u2212₹277 / \u2212₹15"],
    ["Reviews after 4 weeks at ₹129", "26 / 35", "14 / 20", "8 / 11"],
], [52 * mm, (W - 52 * mm) / 3, (W - 52 * mm) / 3, (W - 52 * mm) / 3]))
s.append(Spacer(1, 2))
s.append(P("<b>What it shows:</b> both agree that launching below the floor loses money for every kind of product. The models are more "
           "cautious than the simulator about discounts that stay above the floor: sellers in the data cut prices when sales are slow, "
           "which blurs the price effect. Only a randomised price test removes that, so the pilot includes one.", body))

s.append(P("What is \u201clisting quality\u201d?", h2))
s.append(P("A 0\u2013100 score for <b>how well the product is presented</b>: clear photos, a plain well-lit background, a full description, "
           "a size chart or ingredient list. It is not product quality, which buyers learn after delivery through ratings and returns. "
           "In the app the seller ticks 4 boxes, each worth 7.5 points (score 50 to 80). <b>One extra box (soap):</b> conversion about +47%, "
           "impressions +5%, returns about 11% lower. The conversion effect is large because the score also stands in for product "
           "quality the model can't see; a real version would compute it from Meesho's catalog data and test it.", body))

s.append(P("Which app output comes from which model, and how good it is", h2))
s.append(table([
    ["What the seller sees", "Comes from"],
    ["Floor price and the bill for one paid order", "B only, plus the seller's own costs"],
    ["Start price and good range", "A1 + A2 + B, simulated for 12 weeks at every candidate price"],
    ["Launch lower: reviews bought, cost per review, payback", "A2's sales-to-visibility loop, A1 and B, over 26 weeks"],
    ["\u201cAt 4.1" + STAR + " you can charge about 8% more\u201d", "Rating and price effects from A1 and A2 (demo data)"],
], [W * 0.5, W * 0.5]))
s.append(Spacer(1, 2))
s.append(P("<b>Backtest:</b> hide 20% of listings, retrain, predict their 8-week orders: typical error <b>31% (soap) and 41% (t-shirt)</b>, "
           "against 153% and 87% for a simple average. <b>Limits:</b> synthetic data, two categories, simple regressions chosen so every "
           "number can be explained. <b>For real:</b> train on Meesho's order, return and review history, retrain regularly, and A/B test "
           "every price change.", body))

doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("ok")

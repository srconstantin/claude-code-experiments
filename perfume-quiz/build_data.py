"""Generate data.js from Perfume Quiz.xlsx.

Questions are hand-curated with 5-point option labels.
Answers are extracted as-is from the sheet (keeping the author's original keys).
Known quirks in the sheet (14 duplicate + 14 missing keys) are handled at
runtime via Hamming-distance nearest-neighbor lookup.
"""
import json
import openpyxl

AXES = [
    {"name": "Dark/Light",         "negative": "Dark",      "positive": "Light"},
    {"name": "Warm/Cool",          "negative": "Warm",      "positive": "Cool"},
    {"name": "Sweet/Dry",          "negative": "Sweet",     "positive": "Dry"},
    {"name": "Masculine/Feminine", "negative": "Masculine", "positive": "Feminine"},
    {"name": "Clean/Dirty",        "negative": "Clean",     "positive": "Dirty"},
    {"name": "Simple/Complex",     "negative": "Simple",    "positive": "Complex"},
    {"name": "Sexy/Reserved",      "negative": "Sexy",      "positive": "Reserved"},
    {"name": "Formal/Casual",      "negative": "Formal",    "positive": "Casual"},
]

# For each question: (axis, text, [5 option labels from first-pole to second-pole]).
# Option index 0 → score -2, index 4 → score +2.
# First pole (negative side): Dark / Warm / Sweet / Masculine / Clean / Simple / Sexy / Formal.
QUESTIONS = [
    # === Dark/Light (axis 0) — first pole is Dark ===
    (0, "Do you prefer red or white wine?",
     ["Strongly prefer red", "Slightly prefer red", "Neutral", "Slightly prefer white", "Strongly prefer white"]),
    (0, "Do you like vanilla or chocolate?",
     ["Strongly prefer chocolate", "Slightly prefer chocolate", "Neutral", "Slightly prefer vanilla", "Strongly prefer vanilla"]),
    (0, "Do you like crisp, refreshing cocktails (martini, margarita) or deep, rich ones (rum & coke, old fashioned)?",
     ["Strongly prefer deep & rich", "Slightly prefer deep & rich", "Neutral", "Slightly prefer crisp & refreshing", "Strongly prefer crisp & refreshing"]),
    (0, "Do you prefer light colors or dark colors?",
     ["Strongly prefer dark", "Slightly prefer dark", "Neutral", "Slightly prefer light", "Strongly prefer light"]),
    (0, "Is your aesthetic more moody or more cheery?",
     ["Very moody", "Somewhat moody", "Neutral", "Somewhat cheery", "Very cheery"]),
    (0, "Do you like the smell of smoke?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (0, "Do you like the smell of leather?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (0, "Do you like the smell of citrus?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (0, "Is your aesthetic more \"airy\" or \"earthy\"?",
     ["Very earthy", "Somewhat earthy", "Neutral", "Somewhat airy", "Very airy"]),
    (0, "Would you rather come across as \"cute\" or \"powerful\"?",
     ["Very powerful", "Somewhat powerful", "Neutral", "Somewhat cute", "Very cute"]),
    (0, "Do you like \"darker\" scents or \"lighter\" scents?",
     ["Much darker", "Somewhat darker", "Neutral", "Somewhat lighter", "Much lighter"]),
    (0, "Are you more demonic or more angelic?",
     ["Very demonic", "Somewhat demonic", "Neutral", "Somewhat angelic", "Very angelic"]),
    (0, "Are you more elf or dwarf?",
     ["Very dwarf", "Somewhat dwarf", "Neutral", "Somewhat elf", "Very elf"]),
    (0, "Do you want a scent for day or night?",
     ["Strongly for night", "Slightly for night", "Neutral", "Slightly for day", "Strongly for day"]),
    (0, "Do you like milky/creamy scents?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),
    (0, "Do you like \"powdery\" scents? (think talcum powder)",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),
    (0, "Do you like dark wood (ebony, oak, mahogany) or light wood (pine, cedar)?",
     ["Strongly prefer dark wood", "Slightly prefer dark wood", "Neutral", "Slightly prefer light wood", "Strongly prefer light wood"]),
    (0, "Do you like dark mode or light mode on computers?",
     ["Strongly prefer dark mode", "Slightly prefer dark mode", "Neutral", "Slightly prefer light mode", "Strongly prefer light mode"]),
    (0, "Do you like the smell of tar or gasoline?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),

    # === Warm/Cool (axis 1) — first pole is Warm ===
    (1, "Do you prefer gold or silver jewelry/accessories?",
     ["Strongly prefer gold", "Slightly prefer gold", "Neutral", "Slightly prefer silver", "Strongly prefer silver"]),
    (1, "Do you like warm colors (red, orange, yellow) or cool colors (green, blue, purple)?",
     ["Strongly prefer warm colors", "Slightly prefer warm colors", "Neutral", "Slightly prefer cool colors", "Strongly prefer cool colors"]),
    (1, "Are you more extroverted or more introverted?",
     ["Very extroverted", "Somewhat extroverted", "Neutral", "Somewhat introverted", "Very introverted"]),
    (1, "Would you rather smell warm and cozy, or cool and refreshing?",
     ["Strongly warm & cozy", "Slightly warm & cozy", "Neutral", "Slightly cool & refreshing", "Strongly cool & refreshing"]),
    (1, "Do you like \"browned\" flavors like caramel, maple syrup, molasses, or toasted nuts?",
     ["Love them", "Like them", "Neutral", "Dislike them", "Hate them"]),
    (1, "Do you like the smell of bug spray?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (1, "Do you like the smell of green leaves and herbs?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (1, "Do you like the smell of a forest?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (1, "Do you like the smell of pumpkin spice? (cinnamon, cloves, nutmeg)",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (1, "Do you prefer the smell of fresh fruit or the smell of jam?",
     ["Strongly prefer jam", "Slightly prefer jam", "Neutral", "Slightly prefer fresh fruit", "Strongly prefer fresh fruit"]),
    (1, "Do you prefer fire or water?",
     ["Strongly prefer fire", "Slightly prefer fire", "Neutral", "Slightly prefer water", "Strongly prefer water"]),
    (1, "Do you like licorice as a smell/flavor?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (1, "Do you like mint as a smell/flavor?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (1, "What sounds better, warming up by the fire in winter with a mug of tea/cocoa, or cooling off with a lemonade by the beach in summer?",
     ["Strongly warming up by the fire", "Slightly warming up by the fire", "Neutral", "Slightly cooling off by the beach", "Strongly cooling off by the beach"]),
    (1, "What do you like better, the sun or the moon?",
     ["Strongly prefer the sun", "Slightly prefer the sun", "Neutral", "Slightly prefer the moon", "Strongly prefer the moon"]),
    (1, "Are you more warrior or more magician?",
     ["Very warrior", "Somewhat warrior", "Neutral", "Somewhat magician", "Very magician"]),
    (1, "Do you come across as more friendly or more intimidating?",
     ["Very friendly", "Somewhat friendly", "Neutral", "Somewhat intimidating", "Very intimidating"]),
    (1, "Do you like the smell of camphor? (think tiger balm, Vick's Vap-O-Rub)",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),

    # === Sweet/Dry (axis 2) — first pole is Sweet ===
    (2, "Do you like the smell of desserts?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (2, "Do you prefer sweet scents or unsweet/dry scents?",
     ["Strongly prefer sweet", "Slightly prefer sweet", "Neutral", "Slightly prefer dry", "Strongly prefer dry"]),
    (2, "Do you like the smell of fruit?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (2, "Do you like the smell of vanilla?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (2, "Do you have a sweet tooth?",
     ["Huge sweet tooth", "Somewhat sweet tooth", "Neutral", "Not really", "Not at all"]),
    (2, "Do you like bitter flavors?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),
    (2, "Do you greet people with a hug?",
     ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    (2, "Are you more grouchy or delighted?",
     ["Very delighted", "Somewhat delighted", "Neutral", "Somewhat grouchy", "Very grouchy"]),
    (2, "Do you like mellow scents or sharp ones?",
     ["Strongly prefer mellow", "Slightly prefer mellow", "Neutral", "Slightly prefer sharp", "Strongly prefer sharp"]),

    # === Masculine/Feminine (axis 3) — first pole is Masculine ===
    (3, "Is your aesthetic more masculine or more feminine?",
     ["Strongly masculine", "Slightly masculine", "Neutral", "Slightly feminine", "Strongly feminine"]),
    (3, "Do you like the smell of flowers?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (3, "Do you like the smell of Axe body spray?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (3, "Would you describe yourself as \"macho\"?",
     ["Absolutely", "Somewhat", "Neutral", "Not really", "Not at all"]),
    (3, "Would you describe yourself as \"girly\"?",
     ["Not at all", "Not really", "Neutral", "Somewhat", "Absolutely"]),
    (3, "Would you describe yourself as \"fatherly\"?",
     ["Absolutely", "Somewhat", "Neutral", "Not really", "Not at all"]),
    (3, "Would you describe yourself as \"motherly\"?",
     ["Not at all", "Not really", "Neutral", "Somewhat", "Absolutely"]),
    (3, "Would you describe yourself as \"butch\"?",
     ["Absolutely", "Somewhat", "Neutral", "Not really", "Not at all"]),
    (3, "Do you like \"frou-frou\", frilly, dainty clothes and/or decor?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),
    (3, "Do you resonate more with Venus or Mars?",
     ["Strongly Mars", "Slightly Mars", "Neutral", "Slightly Venus", "Strongly Venus"]),

    # === Clean/Dirty (axis 4) — first pole is Clean ===
    (4, "Do you like the smell of sweat?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (4, "Do you like \"clean\" or \"dirty\" fragrances?",
     ["Strongly prefer clean", "Slightly prefer clean", "Neutral", "Slightly prefer dirty", "Strongly prefer dirty"]),
    (4, "Do you like \"earthy\" smells?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),
    (4, "Do you want a perfume that reminds you of fur?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (4, "Do you like the smell of horses?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (4, "Do you like the smell of soap?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (4, "Do you like the smell of dirt?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (4, "Do you like the smell of freshly laundered sheets?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (4, "Do you like slick, glossy, futuristic aesthetics?",
     ["Love them", "Like them", "Neutral", "Dislike them", "Hate them"]),
    (4, "Do you like natural, rough, outdoorsy aesthetics?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),
    (4, "Do you like blue cheese?",
     ["Hate it", "Dislike it", "Neutral", "Like it", "Love it"]),
    (4, "Do you like \"funky\" smells (think mushrooms or kimchi)?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),

    # === Simple/Complex (axis 5) — first pole is Simple ===
    (5, "Is your aesthetic more minimalist or more ornamented?",
     ["Very minimalist", "Somewhat minimalist", "Neutral", "Somewhat ornamented", "Very ornamented"]),
    (5, "Do you like modern architecture?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (5, "Do you like symphonies and/or operas?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),
    (5, "Is your aesthetic more retro or modern?",
     ["Very modern", "Somewhat modern", "Neutral", "Somewhat retro", "Very retro"]),
    (5, "Do you want a simple perfume or a complex one?",
     ["Strongly simple", "Slightly simple", "Neutral", "Slightly complex", "Strongly complex"]),
    (5, "Do you like trance/ambient/techno music?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (5, "Do you like abstract art?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (5, "Do you like serif or sans serif fonts?",
     ["Strongly prefer sans serif", "Slightly prefer sans serif", "Neutral", "Slightly prefer serif", "Strongly prefer serif"]),
    (5, "Do you like straight lines or complicated curves?",
     ["Strongly prefer straight lines", "Slightly prefer straight lines", "Neutral", "Slightly prefer complicated curves", "Strongly prefer complicated curves"]),
    (5, "Do you like vintage or retro perfumes?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),
    (5, "Do you like wearing a lot of jewelry and/or accessories?",
     ["Hate to", "Rarely", "Neutral", "Often", "Love to"]),
    (5, "Do you like sleek, streamlined aesthetics?",
     ["Love them", "Like them", "Neutral", "Dislike them", "Hate them"]),
    (5, "Do you like elaborate, baroque aesthetics?",
     ["Hate them", "Dislike them", "Neutral", "Like them", "Love them"]),

    # === Sexy/Reserved (axis 6) — first pole is Sexy ===
    (6, "Is your aesthetic more sexy or reserved?",
     ["Very sexy", "Somewhat sexy", "Neutral", "Somewhat reserved", "Very reserved"]),
    (6, "Do you want a perfume for going out at night or for wearing to work/school?",
     ["Strongly for going out at night", "Slightly for going out at night", "Neutral", "Slightly for work/school", "Strongly for work/school"]),
    (6, "Do you like clothes that show off your body?",
     ["Love to wear them", "Often wear them", "Neutral", "Rarely wear them", "Never wear them"]),
    (6, "Would you rather come across as more slutty or more buttoned-up?",
     ["Strongly slutty", "Slightly slutty", "Neutral", "Slightly buttoned-up", "Strongly buttoned-up"]),
    (6, "Is your sexuality blatant or subtle?",
     ["Very blatant", "Somewhat blatant", "Neutral", "Somewhat subtle", "Very subtle"]),
    (6, "Would you describe yourself as \"no-nonsense\"?",
     ["Not at all", "Not really", "Neutral", "Somewhat", "Absolutely"]),
    (6, "Would you describe yourself as \"smoldering\"?",
     ["Absolutely", "Somewhat", "Neutral", "Not really", "Not at all"]),
    (6, "Do you have a \"bad boy\"/\"bad girl\" aesthetic?",
     ["Absolutely", "Somewhat", "Neutral", "Not really", "Not at all"]),
    (6, "Would you describe yourself as \"aloof\"?",
     ["Not at all", "Not really", "Neutral", "Somewhat", "Absolutely"]),
    (6, "Would you rather be hot or interesting?",
     ["Strongly hot", "Slightly hot", "Neutral", "Slightly interesting", "Strongly interesting"]),
    (6, "Would you describe yourself as \"innocent\"?",
     ["Not at all", "Not really", "Neutral", "Somewhat", "Absolutely"]),
    (6, "Would you describe yourself as \"lustful\"?",
     ["Absolutely", "Somewhat", "Neutral", "Not really", "Not at all"]),
    (6, "Do you like perfumes that smell sexy?",
     ["Love them", "Like them", "Neutral", "Dislike them", "Hate them"]),

    # === Formal/Casual (axis 7) — first pole is Formal ===
    (7, "Do you want a more formal or more casual perfume?",
     ["Strongly formal", "Slightly formal", "Neutral", "Slightly casual", "Strongly casual"]),
    (7, "Do you like wearing a suit and/or evening gown?",
     ["Love to", "Often", "Neutral", "Rarely", "Never"]),
    (7, "Is your style more youthful or more mature?",
     ["Very mature", "Somewhat mature", "Neutral", "Somewhat youthful", "Very youthful"]),
    (7, "Do you like to dress up or dress down?",
     ["Strongly dress up", "Slightly dress up", "Neutral", "Slightly dress down", "Strongly dress down"]),
    (7, "Do you like feeling like a \"lady\" and/or \"gentleman\"?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
    (7, "Would you describe yourself as \"classy\"?",
     ["Absolutely", "Somewhat", "Neutral", "Not really", "Not at all"]),
    (7, "Would you describe yourself as \"old-fashioned\"?",
     ["Absolutely", "Somewhat", "Neutral", "Not really", "Not at all"]),
    (7, "Would you describe yourself as \"laid-back\"?",
     ["Not at all", "Not really", "Neutral", "Somewhat", "Absolutely"]),
    (7, "Is etiquette overrated or underrated?",
     ["Very underrated", "Somewhat underrated", "Neutral", "Somewhat overrated", "Very overrated"]),
    (7, "Do you like feeling \"glamorous\"?",
     ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]),
]


def interleave_by_axis(questions):
    """Interleave questions so consecutive ones come from different axes."""
    by_axis = {i: [] for i in range(8)}
    for q in questions:
        by_axis[q[0]].append(q)
    out = []
    while any(by_axis[i] for i in range(8)):
        for i in range(8):
            if by_axis[i]:
                out.append(by_axis[i].pop(0))
    return out


def load_answers():
    wb = openpyxl.load_workbook('Perfume Quiz.xlsx')
    ws = wb['Answers']
    entries = []
    for row in ws.iter_rows(values_only=True):
        if not any(c is not None for c in row):
            continue
        key_raw, name, desc = row[0], row[1], row[2]
        if isinstance(key_raw, float):
            key = str(int(key_raw)).zfill(8)
        elif isinstance(key_raw, str):
            key = key_raw.zfill(8)
        else:
            key = str(key_raw)
        entries.append({"key": key, "name": name, "description": desc})
    return entries


def main():
    # Sanity check coverage: every axis has at least one question
    counts = [0]*8
    for q in QUESTIONS:
        counts[q[0]] += 1
    print(f"Questions per axis: {counts}  (total: {sum(counts)})")

    interleaved = interleave_by_axis(QUESTIONS)
    questions_out = [
        {"axis": axis, "text": text, "options": opts}
        for axis, text, opts in interleaved
    ]

    answers_out = load_answers()
    print(f"Answers: {len(answers_out)}")

    # Build JS data module
    js = []
    js.append("// Auto-generated by build_data.py — do not edit by hand.")
    js.append("")
    js.append("export const AXES = " + json.dumps(AXES, indent=2, ensure_ascii=False) + ";")
    js.append("")
    js.append("export const QUESTIONS = " + json.dumps(questions_out, indent=2, ensure_ascii=False) + ";")
    js.append("")
    js.append("export const ANSWERS = " + json.dumps(answers_out, indent=2, ensure_ascii=False) + ";")
    js.append("")

    with open('data.js', 'w', encoding='utf-8') as f:
        f.write("\n".join(js))
    print("Wrote data.js")


if __name__ == '__main__':
    main()

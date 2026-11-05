"""
Plain-language explanation of a screening result, in English, Hindi or Telugu.

If an Anthropic API key is configured (ANTHROPIC_API_KEY), Claude writes a
short personalised explanation. Otherwise (or if the call fails) a built-in
template is used, so the app always works offline from the API.
"""

from __future__ import annotations

import os

LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu"}

TEMPLATES = {
    "en": {
        "none": "Your eyelid colour looks healthy. The estimated haemoglobin is about {hb} grams per decilitre, which is in the normal range. Keep eating iron-rich foods like green leafy vegetables, lentils, jaggery and eggs.",
        "mild": "Your eyelid looks slightly pale. The estimated haemoglobin is about {hb}, which suggests mild anaemia. Eat more iron-rich foods such as spinach, lentils, dates and jaggery, with lemon or amla to help absorption. Please get a blood test at the health centre within two weeks.",
        "moderate": "Your eyelid looks pale. The estimated haemoglobin is about {hb}, which suggests moderate anaemia. Please visit the health centre this week for a blood test and iron tablets. Avoid tea or coffee with meals, as they reduce iron absorption.",
        "severe": "Your eyelid looks very pale. The estimated haemoglobin is about {hb}, which suggests severe anaemia. This needs medical care soon. Please go to the health centre within one or two days. If you feel breathless, dizzy or have chest pain, go immediately.",
        "disclaimer": "This is a screening estimate, not a diagnosis. Only a blood test can confirm anaemia.",
    },
    "hi": {
        "none": "आपकी पलक का रंग स्वस्थ दिखता है। अनुमानित हीमोग्लोबिन लगभग {hb} ग्राम प्रति डेसीलीटर है, जो सामान्य है। हरी पत्तेदार सब्ज़ियाँ, दालें, गुड़ और अंडे खाते रहें।",
        "mild": "आपकी पलक थोड़ी पीली दिखती है। अनुमानित हीमोग्लोबिन लगभग {hb} है, जो हल्के एनीमिया का संकेत है। पालक, दाल, खजूर और गुड़ जैसे आयरन वाले भोजन ज़्यादा खाएँ, साथ में नींबू या आँवला लें। दो हफ़्ते के अंदर स्वास्थ्य केंद्र पर खून की जाँच कराएँ।",
        "moderate": "आपकी पलक पीली दिखती है। अनुमानित हीमोग्लोबिन लगभग {hb} है, जो मध्यम एनीमिया का संकेत है। कृपया इसी हफ़्ते स्वास्थ्य केंद्र जाकर खून की जाँच और आयरन की गोलियाँ लें। खाने के साथ चाय या कॉफ़ी न पिएँ।",
        "severe": "आपकी पलक बहुत पीली दिखती है। अनुमानित हीमोग्लोबिन लगभग {hb} है, जो गंभीर एनीमिया का संकेत है। एक-दो दिन में स्वास्थ्य केंद्र ज़रूर जाएँ। अगर साँस फूले, चक्कर आए या सीने में दर्द हो तो तुरंत जाएँ।",
        "disclaimer": "यह केवल जाँच का अनुमान है, निदान नहीं। एनीमिया की पुष्टि केवल खून की जाँच से होती है।",
    },
    "te": {
        "none": "మీ కనురెప్ప రంగు ఆరోగ్యంగా కనిపిస్తోంది. అంచనా హిమోగ్లోబిన్ సుమారు {hb} గ్రాములు, ఇది సాధారణ స్థాయి. ఆకుకూరలు, పప్పులు, బెల్లం, గుడ్లు వంటి ఐరన్ ఉన్న ఆహారం తింటూ ఉండండి.",
        "mild": "మీ కనురెప్ప కొంచెం పాలిపోయినట్లు ఉంది. అంచనా హిమోగ్లోబిన్ సుమారు {hb}, ఇది తేలికపాటి రక్తహీనతను సూచిస్తుంది. పాలకూర, పప్పు, ఖర్జూరం, బెల్లం వంటి ఐరన్ ఆహారం ఎక్కువగా తినండి, నిమ్మ లేదా ఉసిరి కూడా తీసుకోండి. రెండు వారాల్లో ఆరోగ్య కేంద్రంలో రక్త పరీక్ష చేయించుకోండి.",
        "moderate": "మీ కనురెప్ప పాలిపోయినట్లు ఉంది. అంచనా హిమోగ్లోబిన్ సుమారు {hb}, ఇది మధ్యస్థ రక్తహీనతను సూచిస్తుంది. ఈ వారంలోనే ఆరోగ్య కేంద్రానికి వెళ్లి రక్త పరీక్ష చేయించుకుని ఐరన్ మాత్రలు తీసుకోండి. భోజనంతో టీ లేదా కాఫీ తాగకండి.",
        "severe": "మీ కనురెప్ప చాలా పాలిపోయినట్లు ఉంది. అంచనా హిమోగ్లోబిన్ సుమారు {hb}, ఇది తీవ్రమైన రక్తహీనతను సూచిస్తుంది. ఒకటి రెండు రోజుల్లో తప్పకుండా ఆరోగ్య కేంద్రానికి వెళ్లండి. ఆయాసం, తల తిరగడం లేదా ఛాతీ నొప్పి ఉంటే వెంటనే వెళ్లండి.",
        "disclaimer": "ఇది కేవలం స్క్రీనింగ్ అంచనా మాత్రమే, నిర్ధారణ కాదు. రక్త పరీక్ష ద్వారా మాత్రమే రక్తహీనత నిర్ధారణ అవుతుంది.",
    },
}

SYSTEM_PROMPT = """You explain anaemia screening results to patients and community health workers in rural areas.
Write 3 to 4 short, simple sentences in {language} that a person with little schooling can understand when read aloud.
Say what the result means, give one or two practical food tips using local, affordable foods, and say when to visit the health centre.
Do not give medicine doses. Do not claim a diagnosis. Plain text only, no lists or markdown."""


def template_explanation(risk: str, hb: float, lang: str) -> dict:
    t = TEMPLATES.get(lang, TEMPLATES["en"])
    return {"text": t[risk].format(hb=hb), "disclaimer": t["disclaimer"], "source": "template"}


def claude_explanation(risk: str, hb: float, lang: str, age: int | None, group: str) -> dict | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(timeout=20.0, max_retries=1)
        prompt = (
            f"Screening result: estimated haemoglobin {hb} g/dL, anaemia risk level '{risk}'. "
            f"Patient group: {group}. Age: {age if age else 'unknown'}."
        )
        response = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT.format(language=LANG_NAMES.get(lang, "English")),
            messages=[{"role": "user", "content": prompt}],
            # Passed via extra_body / extra_headers so this also works on older SDK versions.
            extra_body={"output_config": {"effort": "low"}, "fallbacks": "default"},
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
        )
        if response.stop_reason == "refusal":
            return None
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
        if not text:
            return None
        disclaimer = TEMPLATES.get(lang, TEMPLATES["en"])["disclaimer"]
        return {"text": text, "disclaimer": disclaimer, "source": "claude"}
    except Exception as exc:  # network, auth, rate limit: fall back to template
        print(f"[explain] Claude call failed, using template: {exc}")
        return None


def explain(risk: str, hb: float, lang: str = "en", age: int | None = None, group: str = "woman") -> dict:
    return claude_explanation(risk, hb, lang, age, group) or template_explanation(risk, hb, lang)

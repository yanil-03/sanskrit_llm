# 11. Test Results — What Worked and What Didn't

*A note on method before the examples below: an earlier draft of this file computed the chrF++ figures in 11.1/11.2 by hand for a small set of examples I picked by reading through `eval/report.md` — illustrative, not automated. Since then I added a `score_and_rank()` step (sentence-level `sacrebleu.sentence_chrf` over every prediction in `eval/predictions.jsonl`, sorted), run as a notebook cell after the Section 6 evaluation. The tables below are the real output of that script — every prediction was scored, so these are the genuine top/bottom cases across the full 100-example test set, not a hand-picked sample. (Previously noted as a follow-up in [08_Debugging.md](./08_Debugging.md) Issue 6 — now done, see that file for the updated status.)*

## 11.1 Best Cases (automated ranking, all task types, top 5 by sentence-level chrF++)

| chrF++ | Task | Sanskrit/English Input | What the Model Said |
|---|---|---|---|
| 41.1 | explain | ततस्तत्र प्रविष्टस्य कौसल्याया निवेशनम्। अधिरुह्यापि शयनं बभूव लुलितं मनः॥ | This verse conveys: And thereupon Kausalyā, having entered into her abode, became very sad, and lay down on the bed. |
| 36.1 | explain | तस्यैषा लोकनाथस्य धर्मपत्नी यशस्विनी॥ सीता नाम वरारोहा यां त्वं हर्तुमिच्छसि। | This verse conveys: And that one is the daughter-in-law of the lord of the world, Sītā, who has been married to you. |
| 34.8 | explain | अथ पञ्चवटीं गच्छन्नन्तरा रघुनन्दनः। आससाद महाकायं गृधं भीमपराक्रमम्॥ | This verse conveys: And having gone to the fifth region, Raghu's son, Rāghava, saw Bhīma in his mighty form. |
| 34.7 | explain | कल्यमुत्थाय देवानां कृत्वा पूजां यथाविधि। वन्दितव्यो दशरथः पिता मम जनेश्वरः॥ | This verse conveys: And having performed all the rites for the welfare of the celestials, I have offered my respects to Dasaratha, my father, as one who has been born again. |
| 33.3 | qa | गुहाश्च विविधा घोरा नानामृगगणाकुलाः... (Kinnara/Gandharva abodes passage) | O you of great intelligence, do you take my life and be happy with these men of diverse kinds, who dwell in forests... |

**What this shows, now that it's a real ranking and not a hand-pick:** all five top scores are `explain` and `qa` — tasks where the model only has to paraphrase content already present in the Sanskrit input, rather than generate novel Sanskrit. It stays anchored to the right entities and topic even when finer detail drifts ("overwhelmed with emotion" → "became very sad").

## 11.2 Worst Cases (automated ranking, all task types, bottom 5 by sentence-level chrF++)

| chrF++ | Task | English Input | What the Model Said (Sanskrit) |
|---|---|---|---|
| 9.0 | en2sa | It began to walk at pleasure near Rāma's asylum. Sometimes going, sometimes stopping... | तस्माद् रामेरूपं सहस्रशीघ्रमिह पुष्पका। तदा वनेष्यन्निश्च मधुराः कथयन्निश्च गच्छन्ति॥ |
| 8.9 | en2sa | And presenting him with many a charming fountain, trees will delight Rāma... | तस्माद् राघवं प्रियमनुज्ञानं सर्वे तदा। अपि च महातेजोऽभिषेकः कौशिकः प्रीतिं भवेत्॥ |
| 8.5 | en2sa | Some on the occasions of the Sradha (first) feast their own friends... | तेन त्वां सर्वं प्रियमनुसृत्ते ददर्श च। अभिराज्यामधीयास्तथा भवति ब्रह्मणः॥ |
| 8.4 | en2sa | Khara stationed on his car in the midst of Rākşasas appeared like red bodied one... | क्षरः सहस्राक्षसाय राजा निशाचरो मुनि। कूटमुपास्ते देवी भवत् प्रभावं गच्छति॥ |
| 8.2 | en2sa | And he spent some time in serving the sacrificial fire and his famous sire... | तस्माद् रमपादः प्रीतिं सर्वे तेन चैव शक्तिं। अनुगच्छन्ति यदि कौशिकं गोष्टं जगत् भूतम्॥ |

**What this shows, now that it's a real ranking and not a hand-pick:** every single one of the five worst outputs across the *entire* test set is `en2sa` — this isn't cherry-picking a bad direction, the automated ranking independently confirms it's the worst by a wide margin. And these aren't "close but slightly off" translations — the model produces fluent, well-formed Devanagari verse with little to no real connection to the source text. That's a genuine limitation of the model at this training scale. See [12_Full_Report.md](./12_Full_Report.md) §12.7 for my take on why.

## 11.3 A Real Example From My Run (en2sa, from the actual notebook output)

**Instruction:** Provide a Sanskrit translation of this English sentence.
**Input:** O Śatrughna, arise! Why sleepest you? Bring you at once that lord of the Nisādhas, Guha, Good betide you! he will take the army (over the stream.)
**Reference:** शत्रुघ्नोत्तिष्ठ किं शेषे निषादाधिपतिं गुहम्। शीघ्रमानय भद्रं ते तारयिष्यति वाहिनीम्॥

**Base model** (no fine-tuning) mostly rambled — it opened with "Here is the Sanskrit translation," produced some garbled Devanagari, and then tacked on an English explanation of what it thought it was doing, which is not remotely the expected output format.

**Fine-tuned model:** श्रीमहात्मनः सर्वतो भूतेषु निशाचरे। गुहा यथा ददर्श राजा निसाधस्य महात्मनः॥

The fine-tuned version is a real improvement in *form* — it produces clean, well-formatted Sanskrit verse in the right shape, without any English commentary bleeding in. It's still not semantically accurate to the reference, which is exactly the en2sa weakness this whole report is upfront about. But going from "doesn't even understand what output format is expected" to "produces plausible-looking Sanskrit verse, just not the right one" is a meaningful step, given only 500 training examples.

## 11.4 A Few More Real Before/After Examples (straight from `eval/report.md`)

**Explain — the model actually gets more faithful, not just more fluent**
- Input: `कल्यमुत्थाय देवानां कृत्वा पूजां यथाविधि। वन्दितव्यो दशरथः पिता मम जनेश्वरः॥`
- Reference: Rising early in the morning, adore the deities, and then bow down to the feet of my father Dasaratha, the lord of men.
- Base model: rambled — quoted the verse back, mistranslated it as "let us offer prayers... it is appropriate for my father, who is Janeshwar," then trailed off mid-explanation.
- Fine-tuned model: "And having performed all the rites for the welfare of the celestials, I have offered my respects to Dasaratha, my father, as one who has been born again." Not word-for-word right, but it correctly identifies this is about honoring the father, Dasaratha — the base model didn't even get that far.

**Explain — same story**
- Input: `तस्यैषा लोकनाथस्य धर्मपत्नी यशस्विनी॥ सीता नाम वरारोहा यां त्वं हर्तुमिच्छसि।`
- Reference: This exquisitely beautiful and far-famed Sītā, whom you are about to steal away, is the married wife of that lord of men.
- Base model: completely missed the point — invented a "goddess Yashasvinī" who doesn't exist in this verse, and wrote a paragraph about her supposed connection to Vishnu/Krishna. A confident, fluent, entirely wrong answer.
- Fine-tuned model: "And that one is the daughter-in-law of the lord of the world, Sītā, who has been married to you." Gets the actual subject (Sītā, marriage, "lord of the world") right, even with "wife" vs "daughter-in-law" mixed up.

**sa2en — a case where the base model hallucinated a totally different scene**
- Input: `सात्वमग्नि प्रविश वा स्वयं वा विश दण्डकान्। रज्जु बद्ध्वाथवा कण्ठे नहि तेऽन्यत् परायणम्॥`
- Reference: Enter fire, or dive into Dandaka, or wind a cord round your neck — there's no other way out for you.
- Base model: produced "Fire of Satva or self or VISH Daundika" and then speculated about "a specific deity... associated with Vishnu" — it doesn't understand the sentence at all.
- Fine-tuned model: "And the fire will either enter into his body or else he will be burnt alive; and if it does not do so, then there is no other than Parāyana." Wrong in the details, but it's clearly reading the same verse — fire, an ultimatum, no other recourse.

**qa — where fine-tuning didn't help much**
- Input: `वृत्तदंष्ट्रो महेष्वासः क्वासौ लक्ष्मणपूर्वजः। यदि जीवामि साध्वेनं पश्येयं सीतया सह॥`
- Reference: Where is that elder brother of Lakşmaņa? If I live, I will see him with Sītā.
- Base model: "The verse describes Lakshmana seeing Vasudeva and Krishna together" — wrong characters entirely.
- Fine-tuned model: closer (mentions Lakşmaņa and Sītā correctly) but garbles the actual question being asked. This is one of the qa cases where I'd say the improvement is real but still pretty limited — worth being honest about rather than glossing over.

**sa2en — a longer passage, fine-tuned model tracks the narrative better**
- Input: a 4-line passage about Rāma slaying the Rākṣasa Virādha and consoling Sītā.
- Base model: fabricated a scene involving "Bhimbal," an "army of demons," and "moonlight" — none of which is in the source.
- Fine-tuned model: correctly identifies Rākşasa being slain, Sītā, Lakşmaņa, and the brothers talking about taking shelter in the forest — the gist is right even though some specific lines are invented.

## 11.5 One More Failure, Checked by Hand
**Input:** यत्र नार्यस्तु पूज्यन्ते रमन्ते तत्र देवताः।
**Model said:** "Therein men and women offer worship to the deities."
**What it actually means:** "Where women are honored, there the gods rejoice."

This verse was one of my hand-picked seed test examples, deliberately kept out of training. Its meaning is compressed and idiomatic — the kind of proverb-style Sanskrit that's genuinely hard for a 1.5B model trained on only 500 examples to get right without having seen something very similar before.

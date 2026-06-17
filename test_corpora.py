import json
import unittest
from corpora import (
    Corpus,
    TokenizedCorpus,
    EncipheredCorpus,
    Bitext,
    MixtureOfBitexts,
    CodeswitchedBitext,
    BatchedBitext,
    TokenizedMixtureOfTextAndGoalEncoding,
)
from configure import create_bitexts
from extract_tok_utils import build_fast_align_dict_from_raw
import torch
from torch import tensor
from tokenization import NllbTokenizer
from transformers import AutoModelForSeq2SeqLM


class TestCorpora(unittest.TestCase):
    def test_streaming_bitext(self):
        bitext = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        expected = [
            ("The cat chased the mouse.", "Le chat a poursuivi la souris."),
            ("She reads a book.", "Elle lit un livre."),
            ("They play soccer.", "Ils jouent au football."),
            ("I ate dinner.", "J’ai dîné."),
            ("He drinks coffee.", "Il boit du café."),
            ("We watched a movie.", "Nous avons regardé un film."),
            ("The dog barked at strangers.", "Le chien a aboyé sur des inconnus."),
            ("You wrote a letter.", "Tu as écrit une lettre."),
            ("John opened the door.", "John a ouvert la porte."),
            ("The teacher gave homework.", "Le professeur a donné des devoirs."),
            ("Sarah paints pictures.", "Sarah peint des tableaux."),
            ("The baby kicked the ball.", "Le bébé a frappé le ballon."),
            ("Tom fixed the bike.", "Tom a réparé le vélo."),
            ("Emma baked a cake.", "Emma a fait un gâteau."),
            ("The child drew a star.", "L’enfant a dessiné une étoile."),
            ("My brother broke the window.", "Mon frère a cassé la fenêtre."),
            ("Lisa hugged her friend.", "Lisa a serré son amie dans ses bras."),
            ("Mark answered the question.", "Mark a répondu à la question."),
            ("The chef cooked a meal.", "Le chef a cuisiné un repas."),
            ("They built a house.", "Ils ont construit une maison."),
        ]
        result = [line for line in bitext]
        self.assertEqual(expected, result)

    def test_streaming_bitext(self):
        bitext = MultifileBitext(
            ["test_files/lang1.txt", "test_files/lang1.txt"],
            ["test_files/lang2.txt", "test_files/lang3.txt"],
        )

        expected = [
            ("The cat chased the mouse.", "Le chat a poursuivi la souris."),
            ("She reads a book.", "Elle lit un livre."),
            ("They play soccer.", "Ils jouent au football."),
            ("I ate dinner.", "J’ai dîné."),
            ("He drinks coffee.", "Il boit du café."),
            ("We watched a movie.", "Nous avons regardé un film."),
            ("The dog barked at strangers.", "Le chien a aboyé sur des inconnus."),
            ("You wrote a letter.", "Tu as écrit une lettre."),
            ("John opened the door.", "John a ouvert la porte."),
            ("The teacher gave homework.", "Le professeur a donné des devoirs."),
            ("Sarah paints pictures.", "Sarah peint des tableaux."),
            ("The baby kicked the ball.", "Le bébé a frappé le ballon."),
            ("Tom fixed the bike.", "Tom a réparé le vélo."),
            ("Emma baked a cake.", "Emma a fait un gâteau."),
            ("The child drew a star.", "L’enfant a dessiné une étoile."),
            ("My brother broke the window.", "Mon frère a cassé la fenêtre."),
            ("Lisa hugged her friend.", "Lisa a serré son amie dans ses bras."),
            ("Mark answered the question.", "Mark a répondu à la question."),
            ("The chef cooked a meal.", "Le chef a cuisiné un repas."),
            ("They built a house.", "Ils ont construit une maison."),
            ("The cat chased the mouse.", "Die Katze jagte die Maus."),
            ("She reads a book.", "Sie liest ein Buch."),
            ("They play soccer.", "Sie spielen Fußball."),
            ("I ate dinner.", "Ich habe zu Abend gegessen."),
            ("He drinks coffee.", "Er trinkt Kaffee."),
            ("We watched a movie.", "Wir haben einen Film gesehen."),
            ("The dog barked at strangers.", "Der Hund bellte Fremde an."),
            ("You wrote a letter.", "Du hast einen Brief geschrieben."),
            ("John opened the door.", "John hat die Tür geöffnet."),
            ("The teacher gave homework.", "Der Lehrer gab Hausaufgaben."),
            ("Sarah paints pictures.", "Sarah malt Bilder."),
            ("The baby kicked the ball.", "Das Baby hat den Ball getreten."),
            ("Tom fixed the bike.", "Tom hat das Fahrrad repariert."),
            ("Emma baked a cake.", "Emma hat einen Kuchen gebacken."),
            ("The child drew a star.", "Das Kind hat einen Stern gezeichnet."),
            ("My brother broke the window.", "Mein Bruder hat das Fenster zerbrochen."),
            ("Lisa hugged her friend.", "Lisa hat ihre Freundin umarmt."),
            ("Mark answered the question.", "Mark hat die Frage beantwortet."),
            ("The chef cooked a meal.", "Der Koch hat eine Mahlzeit gekocht."),
            ("They built a house.", "Sie haben ein Haus gebaut."),
        ]
        result = [line for line in bitext]
        self.assertEqual(expected, result)

    def test_mixture_of_bitexts(self):
        bitext1 = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        bitext2 = Bitext("test_files/lang1.txt", "test_files/lang3.txt")
        mix = MixtureOfBitexts(
            {("lang1", "lang2"): bitext1, ("lang1", "lang3"): bitext2}, 3
        )
        batch = mix.next_batch()
        expected1 = (
            ("The cat chased the mouse.", "She reads a book.", "They play soccer."),
            (
                "Le chat a poursuivi la souris.",
                "Elle lit un livre.",
                "Ils jouent au football.",
            ),
            "lang1",
            "lang2",
        )
        expected2 = (
            ("The cat chased the mouse.", "She reads a book.", "They play soccer."),
            (
                "Die Katze jagte die Maus.",
                "Sie liest ein Buch.",
                "Sie spielen Fußball.",
            ),
            "lang1",
            "lang3",
        )
        self.assertIn(batch, [expected1, expected2])

    def test_mixture_of_bitexts2(self):
        text_files = {
            "lang1": "test_files/lang1.txt",
            "lang2": "test_files/lang2.txt",
            "lang3": "test_files/lang3.txt",
        }
        mix = MixtureOfBitexts.create_from_files(
            text_files, [("lang1", "lang2", None), ("lang1", "lang3", None)], 3
        )
        batch = mix.next_batch()
        expected1 = (
            ("The cat chased the mouse.", "She reads a book.", "They play soccer."),
            (
                "Le chat a poursuivi la souris.",
                "Elle lit un livre.",
                "Ils jouent au football.",
            ),
            "lang1",
            "lang2",
        )
        expected2 = (
            ("The cat chased the mouse.", "She reads a book.", "They play soccer."),
            (
                "Die Katze jagte die Maus.",
                "Sie liest ein Buch.",
                "Sie spielen Fußball.",
            ),
            "lang1",
            "lang3",
        )
        self.assertIn(batch, [expected1, expected2])

    def test_mixture_of_bitexts3(self):
        text_files = {
            "lang1": "test_files/lang1.txt",
            "lang2": "test_files/lang2.txt",
            "lang3": "test_files/lang3.txt",
        }
        mix = MixtureOfBitexts.create_from_files(
            text_files,
            [("lang1", "lang2", None), ("lang1", "lang3", None)],
            batch_size=5,
            only_once_thru=True,
        )
        counter = 0
        batch = "not none"
        while batch is not None:
            batch = mix.next_batch()
            if batch is not None:
                counter += 1
        self.assertEqual(counter, 8)

    def test_mixture_of_bitexts4(self):
        text_files = {
            "lang1": "test_files/lang1.txt",
            "lang2": "test_files/lang2.txt",
            "lang3": "test_files/lang3.txt",
        }
        mix = MixtureOfBitexts.create_from_files(
            text_files,
            [("lang1", "lang2", None), ("lang1", "lang3", None)],
            batch_size=2,
            only_once_thru=True,
        )
        next_batch = "not none"
        while next_batch is not None:
            next_batch = mix.next_batch()
            if next_batch is not None:
                batch = next_batch
        expected1 = (
            ("The chef cooked a meal.", "They built a house."),
            ("Le chef a cuisiné un repas.", "Ils ont construit une maison."),
            "lang1",
            "lang2",
        )
        expected2 = (
            ("The chef cooked a meal.", "They built a house."),
            ("Der Koch hat eine Mahlzeit gekocht.", "Sie haben ein Haus gebaut."),
            "lang1",
            "lang3",
        )
        self.assertIn(batch, [expected1, expected2])

    def test_mixture_of_bitexts5(self):
        with open("test_files/example_config.json") as f:
            config = json.load(f)
        mix = MixtureOfBitexts.create_from_config(config, "dev")
        next_batch = mix.next_batch()
        expected_option1 = (
            ("The cat slept.", "She runs fast."),
            ("Le chat a dormi.", "Elle court vite."),
            ("l1-l2", "lang1"),
            ("l1-l2", "lang2"),
        )
        expected_option2 = (
            ("The cat slept.", "She runs fast."),
            ("Die Katze hat geschlafen.", "Sie rennt schnell."),
            ("l1-l3", "lang1"),
            ("l1-l3", "lang3"),
        )
        self.assertIn(next_batch, [expected_option1, expected_option2])

    def test_mixture_of_bitexts_limited_lines1(self):
        text_files = {
            "lang1": "test_files/lang1.txt",
            "lang2": "test_files/lang2.txt",
            "lang3": "test_files/lang3.txt",
        }
        mix = MixtureOfBitexts.create_from_files(
            text_files, [("lang1", "lang2", [4, 7]), ("lang1", "lang3", [14, 17])], 2
        )
        batch = mix.next_batch()
        expected1 = (
            ("He drinks coffee.", "We watched a movie."),
            ("Il boit du café.", "Nous avons regardé un film."),
            "lang1",
            "lang2",
        )
        expected2 = (
            ("The child drew a star.", "My brother broke the window."),
            (
                "Das Kind hat einen Stern gezeichnet.",
                "Mein Bruder hat das Fenster zerbrochen.",
            ),
            "lang1",
            "lang3",
        )
        self.assertIn(batch, [expected1, expected2])

    def test_mixture_of_bitexts_limited_lines2(self):
        bitext1 = Bitext("test_files/lang1.txt", "test_files/lang2.txt", lines=[4, 7])
        bitext2 = Bitext("test_files/lang1.txt", "test_files/lang3.txt", lines=[14, 17])
        mix = MixtureOfBitexts(
            {("lang1", "lang2"): bitext1, ("lang1", "lang3"): bitext2}, 2
        )
        batch = mix.next_batch()
        expected1 = (
            ("He drinks coffee.", "We watched a movie."),
            ("Il boit du café.", "Nous avons regardé un film."),
            "lang1",
            "lang2",
        )
        expected2 = (
            ("The child drew a star.", "My brother broke the window."),
            (
                "Das Kind hat einen Stern gezeichnet.",
                "Mein Bruder hat das Fenster zerbrochen.",
            ),
            "lang1",
            "lang3",
        )
        self.assertIn(batch, [expected1, expected2])

    def test_tokenized_mixture_of_bitexts(self):

        text_files = {
            ("test", "eng"): "test_files/lang1.txt",
            ("test", "fra"): "test_files/lang2.txt",
        }
        lang_codes = {("test", "eng"): "eng_Latn", ("test", "fra"): "fra_Latn"}
        mix = MixtureOfBitexts.create_from_files(
            text_files, [(("test", "eng"), ("test", "fra"), None)], 3
        )
        tokenizer = NllbTokenizer("600M")
        tmob = TokenizedMixtureOfBitexts(mix, tokenizer, lang_codes=lang_codes)
        lang1_batch, lang2_batch, _, _ = tmob.next_batch()
        expected_lang1_token_ids = tensor(
            [
                [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 1],
                [256047, 13710, 18379, 43583, 2299, 248075, 2, 1, 1],
            ]
        )
        expected_lang2_token_ids = tensor(
            [
                [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2],
                [256057, 19945, 6622, 159, 68078, 248075, 2, -100, -100, -100, -100],
                [256057, 21422, 5665, 138, 1166, 96236, 248075, 2, -100, -100, -100],
            ]
        )
        expected_lang1_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]
        )
        expected_lang2_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ]
        )

        self.assertEqual(
            lang1_batch["input_ids"].tolist(), expected_lang1_token_ids.tolist()
        )
        self.assertEqual(
            lang2_batch["input_ids"].tolist(), expected_lang2_token_ids.tolist()
        )

        self.assertEqual(
            lang1_batch["attention_mask"].tolist(), expected_lang1_mask.tolist()
        )
        self.assertEqual(
            lang2_batch["attention_mask"].tolist(), expected_lang2_mask.tolist()
        )

    def test_tokenized_mixture_of_bitexts_truncated(self):
        text_files = {
            ("test", "eng"): "test_files/lang1.txt",
            ("test", "fra"): "test_files/lang2.txt",
        }
        lang_codes = {("test", "eng"): "eng_Latn", ("test", "fra"): "fra_Latn"}
        mix = MixtureOfBitexts.create_from_files(
            text_files, [(("test", "eng"), ("test", "fra"), None)], 3
        )
        tokenizer = NllbTokenizer("600M", max_length=8)
        tmob = TokenizedMixtureOfBitexts(mix, tokenizer, lang_codes=lang_codes)
        lang1_batch, lang2_batch, _, _ = tmob.next_batch()
        expected_lang1_token_ids = tensor(
            [
                [256047, 1617, 7875, 228, 55501, 349, 227879, 2],
                [256047, 11873, 272, 22665, 9, 28487, 248075, 2],
                [256047, 13710, 18379, 43583, 2299, 248075, 2, 1],
            ]
        )
        expected_lang2_token_ids = tensor(
            [
                [256057, 1181, 32779, 9, 170684, 356, 82, 2],
                [256057, 19945, 6622, 159, 68078, 248075, 2, -100],
                [256057, 21422, 5665, 138, 1166, 96236, 248075, 2],
            ]
        )
        expected_lang1_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ]
        )
        expected_lang2_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        self.assertEqual(
            lang1_batch["input_ids"].tolist(), expected_lang1_token_ids.tolist()
        )
        self.assertEqual(
            lang2_batch["input_ids"].tolist(), expected_lang2_token_ids.tolist()
        )
        self.assertEqual(
            lang1_batch["attention_mask"].tolist(), expected_lang1_mask.tolist()
        )
        self.assertEqual(
            lang2_batch["attention_mask"].tolist(), expected_lang2_mask.tolist()
        )

    def test_tokenized_mixture_of_bitexts_w_permutations(self):
        text_files = {
            ("test", "eng"): "test_files/lang1.txt",
            ("test", "fra"): "test_files/lang2.txt",
        }
        lang_codes = {("test", "eng"): "eng_Latn", ("test", "fra"): "fra_Latn"}
        mix = MixtureOfBitexts.create_from_files(
            text_files, [(("test", "eng"), ("test", "fra"), None)], 3
        )
        tokenizer = NllbTokenizer("600M")
        pmap = {("test", "eng"): lambda x: x + 1, ("test", "fra"): lambda x: x + 2}
        tmob = TokenizedMixtureOfBitexts(
            mix, tokenizer, lang_codes=lang_codes, permutation_map=pmap
        )
        lang1_batch, lang2_batch, _, _ = tmob.next_batch()
        expected_lang1_token_ids = tensor(
            [
                [256048, 1618, 7876, 229, 55502, 350, 227880, 248076, 3],
                [256048, 11874, 273, 22666, 10, 28488, 248076, 3, 2],
                [256048, 13711, 18380, 43584, 2300, 248076, 3, 2, 2],
            ]
        )
        expected_lang2_token_ids = tensor(
            [
                [256059, 1183, 32781, 11, 170686, 358, 84, 326, 40286, 248077, 4],
                [256059, 19947, 6624, 161, 68080, 248077, 4, -98, -98, -98, -98],
                [256059, 21424, 5667, 140, 1168, 96238, 248077, 4, -98, -98, -98],
            ]
        )
        expected_lang1_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]
        )
        expected_lang2_mask = tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ]
        )
        self.assertEqual(
            lang1_batch["input_ids"].tolist(), expected_lang1_token_ids.tolist()
        )
        self.assertEqual(
            lang2_batch["input_ids"].tolist(), expected_lang2_token_ids.tolist()
        )
        self.assertEqual(
            lang1_batch["attention_mask"].tolist(), expected_lang1_mask.tolist()
        )
        self.assertEqual(
            lang2_batch["attention_mask"].tolist(), expected_lang2_mask.tolist()
        )

    def test_monotext_with_goal_encoding(self):
        text_files = {
            ("test", "eng"): "test_files/lang1.txt",
            ("test", "fra"): "test_files/lang2.txt",
        }
        lang_codes = {("test", "eng"): "eng_Latn", ("test", "fra"): "fra_Latn"}
        mix = MixtureOfBitexts.create_from_files(
            text_files, [(("test", "eng"), ("test", "fra"), None)], 3
        )
        tokenizer = NllbTokenizer("600M")
        tmob = TokenizedMixtureOfBitexts(
            mix, tokenizer, lang_codes=lang_codes, use_alt_pad_token_for_tgt_lang=False
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
        tmotge = TokenizedMixtureOfTextAndGoalEncoding(tmob, model.model.encoder)
        lang1_sents, lang1, goal_encodings, goal_mask = tmotge.next_batch()
        # print(lang1_sents["input_ids"])
        # print(lang1)
        # print(goal_encodings)

    def test_tokenized_mixture_of_bitexts_w_selective_permutations(self):
        text_files = {
            ("test", "eng"): "test_files/lang1.txt",
            ("test", "fra"): "test_files/lang2.txt",
        }
        lang_codes = {("test", "eng"): "eng_Latn", ("test", "fra"): "fra_Latn"}
        mix = MixtureOfBitexts.create_from_files(
            text_files, [(("test", "eng"), ("test", "fra"), None)], 3
        )
        tokenizer = NllbTokenizer("600M")
        pmap = {("test", "fra"): lambda x: x + 2}
        tmob = TokenizedMixtureOfBitexts(
            mix,
            tokenizer,
            lang_codes=lang_codes,
            permutation_map=pmap,
            permutation_prob=0.9,
        )
        lang1_batch, lang2_batch, _, _ = tmob.next_batch()
        print(lang1_batch)
        print(lang2_batch)


class TestCorporaCodeswitch(unittest.TestCase):

    def test_bitext(self):
        bitext = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        tokenizer = NllbTokenizer("600M")
        tokenized_bitext = TokenizedBitext(bitext, tokenizer, "eng_Latn", "fra_Latn")
        for src, tgt in tokenized_bitext:
            print(f"src: {src}")
            print(f"tgt: {tgt}")

    def test_codeswitched_bitext(self):
        bitext = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        tokenizer = NllbTokenizer("600M")
        tokenized_bitext = TokenizedBitext(bitext, tokenizer, "eng_Latn", "fra_Latn")
        alignments = build_fast_align_dict_from_raw(
            "test_files/lang1.txt",
            "test_files/lang2.txt",
            tokenizer,
            "eng_Latn",
            "fra_Latn",
        )
        randomizer = lambda ls: ls[len(ls) // 2]
        codeswitched_bitext = CodeswitchedBitext(
            tokenized_bitext, alignments, randomizer=randomizer
        )
        cb_iter = iter(codeswitched_bitext)
        toks1, toks2 = next(cb_iter)
        expected1 = [256047, 1617, 9, 170684, 356, 82, 324, 40284, 248075, 2]
        expected2 = [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2]
        self.assertEqual(toks1, expected1)
        self.assertEqual(toks2, expected2)
        toks1, toks2 = next(cb_iter)
        expected1 = [256047, 11873, 6622, 159, 68078, 248075, 2]
        expected2 = [256057, 19945, 6622, 159, 68078, 248075, 2]
        self.assertEqual(toks1, expected1)
        self.assertEqual(toks2, expected2)
        toks1, toks2 = next(cb_iter)
        expected1 = [256047, 13710, 5665, 138, 1166, 96236, 248075, 2]
        expected2 = [256057, 21422, 5665, 138, 1166, 96236, 248075, 2]
        self.assertEqual(toks1, expected1)
        self.assertEqual(toks2, expected2)

    def test_codeswitched_bitext_dont_use_zero(self):
        bitext = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        tokenizer = NllbTokenizer("600M")
        tokenized_bitext = TokenizedBitext(bitext, tokenizer, "eng_Latn", "fra_Latn")
        alignments = build_fast_align_dict_from_raw(
            "test_files/lang1.txt",
            "test_files/lang2.txt",
            tokenizer,
            "eng_Latn",
            "fra_Latn",
        )
        randomizer = lambda ls: 0
        codeswitched_bitext = CodeswitchedBitext(
            tokenized_bitext, alignments, randomizer=randomizer
        )
        cb_iter = iter(codeswitched_bitext)
        toks1, toks2 = next(cb_iter)
        expected1 = [256047, 1181, 7875, 228, 55501, 349, 227879, 248075, 2]
        expected2 = [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2]
        self.assertEqual(toks1, expected1)
        self.assertEqual(toks2, expected2)
        toks1, toks2 = next(cb_iter)
        expected1 = [256047, 19945, 272, 22665, 9, 28487, 248075, 2]
        expected2 = [256057, 19945, 6622, 159, 68078, 248075, 2]
        self.assertEqual(toks1, expected1)
        self.assertEqual(toks2, expected2)
        toks1, toks2 = next(cb_iter)
        expected1 = [256047, 21422, 18379, 43583, 2299, 248075, 2]
        expected2 = [256057, 21422, 5665, 138, 1166, 96236, 248075, 2]
        self.assertEqual(toks1, expected1)
        self.assertEqual(toks2, expected2)

    def test_batched_bitext(self):
        bitext = Bitext("test_files/lang1.txt", "test_files/lang2.txt")
        tokenizer = NllbTokenizer("600M")
        tokenized_bitext = TokenizedBitext(bitext, tokenizer, "eng_Latn", "fra_Latn")
        batched_bitext = BatchedBitext(
            tokenized_bitext, 4, src_pad_token=0, tgt_pad_token=0
        )
        expected1 = tensor(
            [
                [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 0],
                [256047, 13710, 18379, 43583, 2299, 248075, 2, 0, 0],
                [256047, 117, 6337, 109233, 248075, 2, 0, 0, 0],
            ]
        )
        expected2 = tensor(
            [
                [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2],
                [256057, 19945, 6622, 159, 68078, 248075, 2, 0, 0, 0, 0],
                [256057, 21422, 5665, 138, 1166, 96236, 248075, 2, 0, 0, 0],
                [256057, 156, 3, 913, 15931, 1877, 248075, 2, 0, 0, 0],
            ]
        )
        bitext_iter = iter(batched_bitext)
        lang1, lang2 = next(bitext_iter)
        self.assertTrue(torch.equal(lang1, expected1))
        self.assertTrue(torch.equal(lang2, expected2))

    def test_mixture_of_bitexts(self):
        tokenizer = NllbTokenizer("600M")
        bitext1 = BatchedBitext(
            TokenizedBitext(
                Bitext("test_files/lang1.txt", "test_files/lang2.txt"),
                tokenizer,
                "eng_Latn",
                "fra_Latn",
            ),
            4,
        )
        bitext2 = BatchedBitext(
            TokenizedBitext(
                Bitext("test_files/lang1.txt", "test_files/lang3.txt"),
                tokenizer,
                "eng_Latn",
                "deu_Latn",
            ),
            4,
        )
        mix = MixtureOfBitexts(
            {("lang1", "lang2"): bitext1, ("lang1", "lang3"): bitext2},
            only_once_thru=True,
        )
        counter = 0
        for a, b, c, d in mix:
            # print(f"{c}: {a}")
            # print(f"{d}: {b}")
            counter += 1
        self.assertEqual(counter, 10)


class TestCorporaRevised(unittest.TestCase):

    def test_corpus1(self):
        corpus = Corpus("test_files/lang1.txt")
        corpus_iter = iter(corpus)
        line = next(corpus_iter)
        self.assertEqual(line, "The cat chased the mouse.")
        line = next(corpus_iter)
        self.assertEqual(line, "She reads a book.")
        line = next(corpus_iter)
        self.assertEqual(line, "They play soccer.")

    def test_corpus2(self):
        corpus = Corpus("test_files/lang1.txt")
        counter = 0
        for _ in corpus:
            counter += 1
        self.assertEqual(counter, 20)

    def test_tokenized_corpus(self):
        tokenizer = NllbTokenizer("600M")
        corpus = Corpus("test_files/lang1.txt")
        tokenized_corpus = TokenizedCorpus(corpus, tokenizer, lang_code="eng_Latn")
        corpus_iter = iter(tokenized_corpus)
        tokens = next(corpus_iter)
        self.assertEqual(
            tokens, [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2]
        )
        tokens = next(corpus_iter)
        self.assertEqual(tokens, [256047, 11873, 272, 22665, 9, 28487, 248075, 2])
        tokens = next(corpus_iter)
        self.assertEqual(tokens, [256047, 13710, 18379, 43583, 2299, 248075, 2])

    def test_enciphered_corpus(self):
        tokenizer = NllbTokenizer("600M")
        corpus = Corpus("test_files/lang1.txt")
        tokenized_corpus = TokenizedCorpus(corpus, tokenizer, lang_code="eng_Latn")
        encipherment_dict = {x: x + 5 for x in range(4, 256000)}
        encipherment = lambda x: encipherment_dict.get(x, x)
        enciphered_corpus = EncipheredCorpus(tokenized_corpus, encipherment)
        corpus_iter = iter(enciphered_corpus)
        tokens = next(corpus_iter)
        self.assertEqual(
            tokens, [256047, 1622, 7880, 233, 55506, 354, 227884, 248080, 2]
        )
        tokens = next(corpus_iter)
        self.assertEqual(tokens, [256047, 11878, 277, 22670, 14, 28492, 248080, 2])
        tokens = next(corpus_iter)
        self.assertEqual(tokens, [256047, 13715, 18384, 43588, 2304, 248080, 2])

    def test_bitext(self):
        tokenizer = NllbTokenizer("600M")
        corpus1 = Corpus("test_files/lang1.txt")
        tokenized_corpus1 = TokenizedCorpus(corpus1, tokenizer, lang_code="eng_Latn")
        corpus2 = Corpus("test_files/lang2.txt")
        tokenized_corpus2 = TokenizedCorpus(corpus2, tokenizer, lang_code="fra_Latn")
        bitext = Bitext(tokenized_corpus1, tokenized_corpus2)
        corpus_iter = iter(bitext)
        tokens1, tokens2 = next(corpus_iter)
        self.assertEqual(
            tokens1, [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2]
        )
        self.assertEqual(
            tokens2, [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2]
        )
        tokens1, tokens2 = next(corpus_iter)
        self.assertEqual(tokens1, [256047, 11873, 272, 22665, 9, 28487, 248075, 2])
        self.assertEqual(tokens2, [256057, 19945, 6622, 159, 68078, 248075, 2])

    def test_batched_bitext(self):
        tokenizer = NllbTokenizer("600M")
        corpus1 = Corpus("test_files/lang1.txt")
        tokenized_corpus1 = TokenizedCorpus(corpus1, tokenizer, lang_code="eng_Latn")
        corpus2 = Corpus("test_files/lang2.txt")
        tokenized_corpus2 = TokenizedCorpus(corpus2, tokenizer, lang_code="fra_Latn")
        bitext = Bitext(tokenized_corpus1, tokenized_corpus2)
        batched_bitext = BatchedBitext(bitext, 4, src_pad_token=0, tgt_pad_token=-100)
        expected1 = tensor(
            [
                [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 0],
                [256047, 13710, 18379, 43583, 2299, 248075, 2, 0, 0],
                [256047, 117, 6337, 109233, 248075, 2, 0, 0, 0],
            ]
        )
        expected2 = tensor(
            [
                [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2],
                [256057, 19945, 6622, 159, 68078, 248075, 2, -100, -100, -100, -100],
                [256057, 21422, 5665, 138, 1166, 96236, 248075, 2, -100, -100, -100],
                [256057, 156, 3, 913, 15931, 1877, 248075, 2, -100, -100, -100],
            ]
        )
        bitext_iter = iter(batched_bitext)
        lang1, lang2 = next(bitext_iter)
        self.assertTrue(torch.equal(lang1["input_ids"], expected1))
        self.assertTrue(torch.equal(lang2["input_ids"], expected2))

    def test_mixture_of_bitexts(self):
        tokenizer = NllbTokenizer("600M")
        corpus1 = Corpus("test_files/lang1.txt")
        tokenized_corpus1 = TokenizedCorpus(corpus1, tokenizer, lang_code="eng_Latn")
        corpus2 = Corpus("test_files/lang2.txt")
        tokenized_corpus2 = TokenizedCorpus(corpus2, tokenizer, lang_code="fra_Latn")
        bitext = Bitext(tokenized_corpus1, tokenized_corpus2)
        batched_bitext1 = BatchedBitext(bitext, 4, src_pad_token=0, tgt_pad_token=-100)
        corpus1 = Corpus("test_files/lang1.txt")
        tokenized_corpus1 = TokenizedCorpus(corpus1, tokenizer, lang_code="eng_Latn")
        corpus2 = Corpus("test_files/lang3.txt")
        tokenized_corpus2 = TokenizedCorpus(corpus2, tokenizer, lang_code="fra_Latn")
        bitext = Bitext(tokenized_corpus1, tokenized_corpus2)
        batched_bitext2 = BatchedBitext(bitext, 4, src_pad_token=0, tgt_pad_token=-100)
        bitext1_metadata = {
            "lang1_tokenizer": "nllb",
            "lang1_encipherment": 0,
            "lang1_code": "eng_Latn",
            "lang2_tokenizer": "nllb",
            "lang2_encipherment": 0,
            "lang2_code": "fra_Latn",
        }
        bitext2_metadata = {
            "lang1_tokenizer": "nllb",
            "lang1_encipherment": 0,
            "lang1_code": "eng_Latn",
            "lang2_tokenizer": "nllb",
            "lang2_encipherment": 0,
            "lang2_code": "deu_Latn",
        }
        mix = MixtureOfBitexts(
            {("lang1", "lang2"): batched_bitext1, ("lang1", "lang3"): batched_bitext2},
            {
                ("lang1", "lang2"): bitext1_metadata,
                ("lang1", "lang3"): bitext2_metadata,
            },
            only_once_thru=True,
        )
        counter = 0
        for _ in mix:
            counter += 1
        self.assertEqual(counter, 10)

    def test_mixture_of_bitexts_from_config(self):
        with open("test_files/example_config.json") as f:
            config = json.load(f)
        mix = create_bitexts(config)
        counter = 0
        for _ in mix["dev"]:
            counter += 1
        self.assertEqual(counter, 4)

    def test_mixture_of_bitexts_from_config2(self):
        with open("test_files/example_config.json") as f:
            config = json.load(f)
        mix = create_bitexts(config)
        mix_iter = iter(mix["train"])
        batch1, batch2, metadata = next(mix_iter)
        bitext1_metadata = {
            "lang1_tokenizer": "nllb",
            "lang1_encipherment": 0,
            "lang1_code": "eng_Latn",
            "lang2_tokenizer": "nllb",
            "lang2_encipherment": 0,
            "lang2_code": "fra_Latn",
        }
        bitext2_metadata = {
            "lang1_tokenizer": "nllb",
            "lang1_encipherment": 0,
            "lang1_code": "eng_Latn",
            "lang2_tokenizer": "nllb",
            "lang2_encipherment": 0,
            "lang2_code": "deu_Latn",
        }
        if metadata["lang2_code"] == "fra_Latn":
            expected1 = tensor(
                [
                    [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                    [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 0],
                ]
            )
            expected2 = tensor(
                [
                    [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2],
                    [
                        256057,
                        19945,
                        6622,
                        159,
                        68078,
                        248075,
                        2,
                        -100,
                        -100,
                        -100,
                        -100,
                    ],
                ]
            )
            self.assertEqual(metadata, bitext1_metadata)
        else:
            expected1 = tensor(
                [
                    [256047, 7007, 158826, 349, 9715, 248075, 2, 0],
                    [256047, 1617, 143207, 43982, 13003, 19836, 248075, 2],
                ]
            )
            expected2 = tensor(
                [
                    [256042, 7007, 2658, 566, 39380, 664, 216976, 2799, 248075, 2],
                    [256042, 6856, 33887, 10184, 24453, 211091, 4108, 77678, 248075, 2],
                ]
            )
            self.assertEqual(metadata, bitext2_metadata)
        self.assertTrue(torch.equal(batch1["input_ids"], expected1))
        self.assertTrue(torch.equal(batch2["input_ids"], expected2))


if __name__ == "__main__":
    unittest.main()

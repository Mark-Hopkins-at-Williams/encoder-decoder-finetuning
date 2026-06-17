from compressor import compress, get_predictions

import unittest
from torch import tensor
import torch
import re


class MockAutocompleteModel:
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def eval(self):
        pass

    def __call__(self, input_ids):
        preds = torch.clamp((input_ids + 1) % len(self.alphabet), min=1)
        logits = torch.zeros(preds.shape[0], preds.shape[1], len(self.alphabet))
        logits = logits.scatter(dim=2, index=preds.unsqueeze(-1), value=1)
        return logits


def strings_to_ids(input_strings, letter_to_id):
    rows = []
    for input_string in input_strings:
        row = tensor([letter_to_id[letter] for letter in input_string])
        rows.append(row)
    return torch.stack(rows)


class TestAutocomplete(unittest.TestCase):

    def test_compress_long(self):
        alphabet = "_abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        full_strings = [
            "abcabc",
            "bcabca",
            "abcabc",
            "abcbbc",
            "abcabb",
            "accccc",
            "ccccca",
            "ccccca",
        ]
        input_strings = [f[:-1] for f in full_strings]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)
        # print(f"Inputs                  = {input_strings}")
        # print(
        #     f"Preds                   = {self.logits_to_strings(model(input_ids), alphabet)}"
        # )
        target_strings = [f[1:] for f in full_strings]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(
            model,
            input_ids,
            target_ids,
            prediction_threshold=0.3,
            output_style="-long",
            prediction_mode="top_pred",
        )
        # print(f"Full                    = {full_strings}")
        # print(f"Condensed rep (decoded) = {self.decode_hex_list(text)}")
        received = self.decode_hex_list(text)
        expected = [
            "a😀😀😀😀😀",
            "b😀😀😀😀😀",
            "a😀😀😀😀😀",
            "a😀😀bb😀",
            "a😀😀😀😀b",
            "accccc",
            "ccccc😀",
            "ccccc😀",
        ]
        self.assertEqual(received, expected)

        """
        abc -> bca is perfect
        abc -> abc is 100% wrong
        abc -> bba is wrong right wrong
        """

    def test_compress_short(self):
        alphabet = "_abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        full_strings = [
            "abcabc",
            "bcabca",
            "abcabc",
            "abcbbc",
            "abcabb",
            "accccc",
            "ccccca",
            "ccccca",
        ]
        input_strings = [f[:-1] for f in full_strings]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)

        # print(f"Inputs                  = {input_strings}")
        # print(
        #     f"Preds                   = {self.logits_to_strings(model(input_ids), alphabet)}"
        # )

        target_strings = [f[1:] for f in full_strings]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(
            model,
            input_ids,
            target_ids,
            prediction_threshold=0.45,
            output_style="-short",
            prediction_mode="top_pred",
        )
        # print(f"Condensed rep = {text}")
        # print(f"Full                    = {full_strings}")

        # print(f"Condensed rep (decoded) = {self.decode_hex_list(text)}")
        received = self.decode_hex_list(text)
        expected = [
            "a😀",
            "b😀",
            "a😀",
            "a😀bb😀",
            "a😀b",
            "accccc",
            "ccccc😀",
            "ccccc😀",
        ]
        self.assertEqual(received, expected)

        """
        abc -> bca is perfect
        abc -> abc is 100% wrong
        abc -> bba is wrong right wrong
        """

    def test_compress_dummy(self):
        alphabet = "_abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        full_strings = [
            "abcabb",
            "accccc",
            "ccccca",
            "ccccca",
        ]
        input_strings = [f[:-1] for f in full_strings]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)

        print()
        print(f"Full                    = {full_strings}")
        print(f"Inputs                  = {input_strings}")
        print(
            f"Preds                   = {self.logits_to_strings(model(input_ids), alphabet)}"
        )

        target_strings = [f[1:] for f in full_strings]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(
            model,
            input_ids,
            target_ids,
            prediction_threshold=0.45,
            output_style="-long",
            prediction_mode="top_pred",
        )

        print(f"Condensed rep (decoded) = {self.decode_hex_list(text)}")

        received = self.decode_hex_list(text)
        expected = ["a😀😀😀😀b", "accccc", "ccccc😀", "ccccc😀"]
        self.assertEqual(received, expected)

    def test_threshold_high(self):
        alphabet = "_abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        full_strings = [
            "abcabc",
            "cabcab",
        ]
        input_strings = [f[:-1] for f in full_strings]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)

        print(f"Inputs                  = {input_strings}")
        print(
            f"Preds                   = {self.logits_to_strings(model(input_ids), alphabet)}"
        )

        target_strings = [f[1:] for f in full_strings]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(
            model,
            input_ids,
            target_ids,
            prediction_threshold=0.999,
            output_style="-short",
            prediction_mode="top_pred",
        )
        print(f"Full                    = {full_strings}")

        print(f"Condensed rep (decoded) = {self.decode_hex_list(text)}")
        received = self.decode_hex_list(text)
        expected = ["abcabc", "cabcab"]
        self.assertEqual(received, expected)

    def test_threshold_low(self):
        alphabet = "_abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        full_strings = [
            "abcabc",
            "cabcab",
        ]
        input_strings = [f[:-1] for f in full_strings]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)

        print(f"Inputs                  = {input_strings}")
        print(
            f"Preds                   = {self.logits_to_strings(model(input_ids), alphabet)}"
        )

        target_strings = [f[1:] for f in full_strings]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(
            model,
            input_ids,
            target_ids,
            prediction_threshold=0.2,
            output_style="-short",
            prediction_mode="top_pred",
        )
        print(f"Full                    = {full_strings}")

        print(f"Condensed rep (decoded) = {self.decode_hex_list(text)}")
        received = self.decode_hex_list(text)
        expected = ["a😀", "c😀"]
        self.assertEqual(received, expected)

    def decode_hex_list(self, input_list):
        decoded_result = []
        for line in input_list:
            cleaned = (
                line.replace(r"\x01", "a").replace(r"\x02", "b").replace(r"\x03", "c")
            )

            parts = re.split(r"(😀)", cleaned)

            # Filter out empty strings from the split
            filtered_parts = [p for p in parts if p]

            if filtered_parts:
                decoded_result.append("".join(filtered_parts))

        return decoded_result

    def logits_to_strings(self, logits_tensor, alphabet):

        # Find the index of the max value (1.0) along the last dimension
        indices = torch.argmax(logits_tensor, dim=2)

        decoded_strings = []
        for row in indices:
            # Convert indices back to characters using the string index
            text = "".join([alphabet[i] for i in row])
            decoded_strings.append(text)

        return decoded_strings

    def test_compress_store_num_chars_autocompleted(self):
        alphabet = "_abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        full_strings = [
            "abcabc",
            "bcabca",
            "abcabc",
            "abcbbc",
            "abcabb",
            "accccc",
            "ccccca",
            "ccccca",
        ]
        input_strings = [f[:-1] for f in full_strings]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)
        target_strings = [f[1:] for f in full_strings]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(
            model,
            input_ids,
            target_ids,
            prediction_threshold=0.3,
            prediction_mode="top_pred",
            autocomplete_mode="store_num_chars_autocompleted",
        )
        received = self.decode_hex_list(text)
        ac = [chr(ord("😀") + k) for k in range(10)]
        expected = [
            "a" + ac[5],
            "b" + ac[5],
            "a" + ac[5],
            "a" + ac[2] + "bb" + ac[1],
            "a" + ac[4] + "b",
            "accccc",
            "ccccc" + ac[1],
            "ccccc" + ac[1],
        ]
        self.assertEqual(received, expected)

    def test_compress_no_autocomplete_chars(self):
        alphabet = "_abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        full_strings = [
            "abcabc",
            "bcabca",
            "abcabc",
            "abcbbc",
            "abcabb",
            "accccc",
            "ccccca",
            "ccccca",
        ]
        input_strings = [f[:-1] for f in full_strings]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)
        target_strings = [f[1:] for f in full_strings]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(
            model,
            input_ids,
            target_ids,
            prediction_threshold=0.3,
            prediction_mode="top_pred",
            autocomplete_mode="no_autocomplete_chars",
        )
        received = self.decode_hex_list(text)
        expected = [
            "a",
            "b",
            "a",
            "abb",
            "ab",
            "accccc",
            "ccccc",
            "ccccc",
        ]
        self.assertEqual(received, expected)

    def test_compress_two_types_english_chars(self):

        alphabet = "_abc"
        letter_to_id = {letter: i for (i, letter) in enumerate(alphabet)}
        full_strings = [
            "abcabc",
            "bcabca",
            "abcabc",
            "abcbbc",
            "abcabb",
            "accccc",
            "ccccca",
            "ccccca",
        ]
        stylized_letters = {chr(1): "ą", chr(2): "ḃ", chr(3): "č"}
        input_strings = [f[:-1] for f in full_strings]
        input_ids = strings_to_ids(input_strings, letter_to_id)
        model = MockAutocompleteModel(alphabet)
        target_strings = [f[1:] for f in full_strings]
        target_ids = strings_to_ids(target_strings, letter_to_id)
        text = compress(
            model,
            input_ids,
            target_ids,
            prediction_threshold=0.3,
            prediction_mode="top_pred",
            autocomplete_mode="two_types_english_chars",
            stylized_letters=stylized_letters,
        )
        received = self.decode_hex_list(text)

        expected = [
            "a",
            "b",
            "a",
            f"a{stylized_letters[chr(2)]}b",
            "ab",
            "a" + (stylized_letters[chr(3)] * 4) + "c",
            "c" + (stylized_letters[chr(3)] * 3) + "c",
            "c" + (stylized_letters[chr(3)] * 3) + "c",
        ]
        self.assertEqual(received, expected)


class TestPredictions(unittest.TestCase):

    def test_get_predictions_top_pred(self):
        logits = tensor([[[1.2, 1.5, 0.3], [-0.4, 0.6, 1.0]]])
        preds = get_predictions(logits, prediction_threshold=0.5, mode="top_pred")
        expected = tensor([[-1, 2]])
        self.assertTrue(torch.equal(preds, expected))

    def test_get_predictions_top_pred2(self):
        logits = tensor([[[1.2, 1.5, 0.3], [-0.4, 0.6, 1.0]]])
        preds = get_predictions(logits, prediction_threshold=0.4, mode="top_pred")
        expected = tensor([[1, 2]])
        self.assertTrue(torch.equal(preds, expected))

    def test_get_predictions_top_pred3(self):
        logits = tensor([[[1.2, 1.5, 0.3], [-0.4, 0.6, 1.0]]])
        preds = get_predictions(logits, prediction_threshold=0.6, mode="top_pred")
        expected = tensor([[-1, -1]])
        self.assertTrue(torch.equal(preds, expected))

    def test_get_predictions_margin(self):
        logits = tensor([[[1.2, 1.5, 0.3], [-0.4, 0.6, 1.0]]])
        preds = get_predictions(logits, prediction_threshold=0.15, mode="margin")
        expected = tensor([[-1, 2]])
        self.assertTrue(torch.equal(preds, expected))

    def test_get_predictions_margin2(self):
        logits = tensor([[[1.2, 1.5, 0.3], [-0.4, 0.6, 1.0]]])
        preds = get_predictions(logits, prediction_threshold=0.1, mode="margin")
        expected = tensor([[1, 2]])
        self.assertTrue(torch.equal(preds, expected))

    def test_get_predictions_margin3(self):
        logits = tensor([[[1.2, 1.5, 0.3], [-0.4, 0.6, 1.0]]])
        preds = get_predictions(logits, prediction_threshold=0.9, mode="margin")
        expected = tensor([[-1, -1]])
        self.assertTrue(torch.equal(preds, expected))

    def test_get_predictions_top_pred_batch(self):
        logits = tensor(
            [[[1.2, 1.5, 0.3], [-0.4, 0.6, 1.0]], [[0.8, -0.5, -0.3], [0.4, 0.8, 0.4]]]
        )
        preds = get_predictions(logits, prediction_threshold=0.5, mode="top_pred")
        expected = tensor([[-1, 2], [0, -1]])
        self.assertTrue(torch.equal(preds, expected))


if __name__ == "__main__":
    unittest.main()

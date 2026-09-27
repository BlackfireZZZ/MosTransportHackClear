"""Boundary checks for the standalone historical traffic collector."""

import unittest

from collect_moscow_traffic_archive import parse_article, search_rows

URL = "https://www.mskagency.ru/materials/1"


def article(
    title="Пробки на дорогах Москвы достигли восьми баллов",
    body="Пробки на дорогах Москвы достигли восьми баллов, следует из данных Яндекс Пробки.",
    stamp="01.01.2025 00:01",
):
    return f'<p class="date">{stamp}</p><h1>{title}</h1><div itemprop="articleBody"><p>{body}</p></div>'


class TrafficArchiveTests(unittest.TestCase):
    def test_observation_and_timezone(self):
        row = parse_article(article(), URL)
        self.assertEqual(row["congestion_score"], 8)
        self.assertEqual(row["published_at"], "2025-01-01T00:01:00+03:00")
        self.assertEqual(row["score_provider"], "yandex")
        self.assertIsNone(row["mean_speed_kmh"])

    def test_future_forecast_in_body_does_not_replace_current_score(self):
        row = parse_article(
            article(
                body="Пробки достигли восьми баллов по данным Яндекс. К 20:00 ожидается снижение до четырех баллов."
            ),
            URL,
        )
        self.assertEqual(row["congestion_score"], 8)

    def test_reject_forecast_wrong_geography_and_ambiguous_values(self):
        for title in [
            "Пробки в Москве завтра достигнут восьми баллов",
            "Пробки в Подмосковье достигли восьми баллов",
            "Пробки в Москве могут достигнуть восьми баллов",
            "Пробки в Москве составляют от семи до восьми баллов",
            "Пробки в Москве достигли восемнадцати баллов",
            "Пробки в Москве достигнут восьми баллов к вечеру",
            "Пробки в центре Москвы достигли восьми баллов",
            "Пробки в Москве не достигли восьми баллов",
            "Пробки в Москве достигли почти восьми баллов",
        ]:
            with self.subTest(title=title):
                self.assertIsNone(parse_article(article(title=title), URL))

    def test_reject_mismatched_lead_and_forecast_lead(self):
        for body in [
            "Пробки достигли семи баллов по данным Яндекс.",
            "Ожидается, что пробки достигнут восьми баллов.",
        ]:
            self.assertIsNone(parse_article(article(body=body), URL))

    def test_cutoff(self):
        self.assertIsNotNone(parse_article(article(stamp="31.10.2025 23:59"), URL))
        for stamp in ["01.11.2025 00:00", "31.12.2024 23:59"]:
            self.assertIsNone(parse_article(article(stamp=stamp), URL))

    def test_speed_is_current_and_unambiguous(self):
        body = "Пробки достигли восьми баллов по данным Яндекс. Средняя скорость транспортного потока в Москве составляет 28 км/ч."
        self.assertEqual(parse_article(article(body=body), URL)["mean_speed_kmh"], 28)
        for tail in [
            " Вчера средняя скорость движения составляет 40 км/ч.",
            " Ожидается, что средняя скорость движения составляет 40 км/ч.",
        ]:
            self.assertEqual(
                parse_article(article(body=body + tail), URL)["mean_speed_kmh"], 28
            )
        self.assertIsNone(
            parse_article(
                article(
                    body=body
                    + " Средняя скорость движения в Москве составляет 40 км/ч."
                ),
                URL,
            )["mean_speed_kmh"]
        )

    def test_nested_article_and_local_speed(self):
        raw = (
            article()
            .replace("<p>Пробки", "<div>Photo credit</div><div><p>Пробки")
            .replace("</p></div>", "</p></div></div>")
        )
        self.assertEqual(parse_article(raw, URL)["congestion_score"], 8)
        body = "Пробки достигли восьми баллов по данным Яндекс. Средняя скорость движения на МКАД составляет 70 км/ч."
        self.assertIsNone(parse_article(article(body=body), URL)["mean_speed_kmh"])

    def test_structure_failure_is_not_empty_success(self):
        with self.assertRaises(ValueError):
            parse_article("<html>Access denied</html>", URL)

    def test_search_only_article_entries(self):
        raw = '<li data-material_id="123" data-datei="20250520"><a href="/materials/123" title="Пробки" class="js-title">title</a></li><a href="/materials/999">unrelated</a>'
        self.assertEqual(search_rows(raw), [("123", "20250520", "Пробки")])


if __name__ == "__main__":
    unittest.main()

from .base import IdentifiableCardRect, CardGroup, Deck

SUITED_CARDS = CardGroup(
    card_names=frozenset(
        f"{suit}_{value}"
        for suit in ("hearts", "clubs", "spades", "diamonds")
        for value in list(range(1, 11)) + ["V", "C", "D", "R"]
    ),
    identifiable_rects=frozenset(
        (
            IdentifiableCardRect(
                left_mm=3,
                right_mm=11,
                top_mm=3,
                bottom_mm=21,
                hull_area_range_mm=(50, 80),
            ),
            IdentifiableCardRect(
                left_mm=-3,
                right_mm=-11,
                top_mm=-3,
                bottom_mm=-21,
                hull_area_range_mm=(60, 80),
            ),
        )
    ),
)

ONE_DIGIT_TRUMP_CARDS = CardGroup(
    card_names=frozenset(str(i) for i in range(1, 10)),
    identifiable_rects=frozenset(
        (
            IdentifiableCardRect(
                left_mm=5,
                right_mm=22,
                top_mm=5,
                bottom_mm=24,
                hull_area_range_mm=(110, 130),
            ),
            IdentifiableCardRect(
                left_mm=-5,
                right_mm=-22,
                top_mm=-5,
                bottom_mm=-24,
                hull_area_range_mm=(110, 130),
            ),
        )
    ),
)

TWO_DIGIT_TRUMP_CARDS = CardGroup(
    card_names=frozenset(str(i) for i in range(10, 22)),
    identifiable_rects=frozenset(
        (
            IdentifiableCardRect(
                left_mm=5,
                right_mm=22,
                top_mm=5,
                bottom_mm=24,
                hull_area_range_mm=(200, 240),
            ),
            IdentifiableCardRect(
                left_mm=-5,
                right_mm=-22,
                top_mm=-5,
                bottom_mm=-24,
                hull_area_range_mm=(200, 240),
            ),
        )
    ),
)

FOOL = CardGroup(
    card_names=frozenset(("fool",)),
    identifiable_rects=frozenset(
        (
            IdentifiableCardRect(
                left_mm=4,
                right_mm=12,
                top_mm=4,
                bottom_mm=13,
                hull_area_range_mm=(16, 25),
            ),
            IdentifiableCardRect(
                left_mm=-4,
                right_mm=-12,
                top_mm=-4,
                bottom_mm=-13,
                hull_area_range_mm=(16, 25),
            ),
        )
    ),
)

DECK = Deck(
    width_mm=66,
    height_mm=120,
    cards=frozenset(
        (
            SUITED_CARDS,
            ONE_DIGIT_TRUMP_CARDS,
            TWO_DIGIT_TRUMP_CARDS,
            FOOL,
        )
    ),
)

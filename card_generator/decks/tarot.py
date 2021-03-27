from .base import CardRect, CardGroup, Deck

SUITED_CARDS = CardGroup(
    card_names=frozenset(
        f"{suit}_{value}"
        for suit in ("hearts", "clubs", "spades", "diamonds")
        for value in list(range(1, 11)) + ["V", "C", "D", "R"]
    ),
    identifiable_rects=frozenset(
        (
            CardRect(
                left_mm=3,
                right_mm=11,
                top_mm=3,
                bottom_mm=21,
            ),
            CardRect(
                left_mm=-3,
                right_mm=-11,
                top_mm=-3,
                bottom_mm=-21,
            ),
        )
    ),
)

TRUMP_CARDS = CardGroup(
    card_names=frozenset(str(i) for i in range(1, 22)),
    identifiable_rects=frozenset(
        (
            CardRect(
                left_mm=5,
                right_mm=22,
                top_mm=5,
                bottom_mm=24,
            ),
            CardRect(
                left_mm=-5,
                right_mm=-22,
                top_mm=-5,
                bottom_mm=-24,
            ),
        )
    ),
)

FOOL = CardGroup(
    card_names=frozenset(("fool",)),
    identifiable_rects=frozenset(
        (
            CardRect(
                left_mm=4,
                right_mm=12,
                top_mm=4,
                bottom_mm=13,
            ),
            CardRect(
                left_mm=-4,
                right_mm=-12,
                top_mm=-4,
                bottom_mm=-13,
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
            TRUMP_CARDS,
            FOOL,
        )
    ),
)

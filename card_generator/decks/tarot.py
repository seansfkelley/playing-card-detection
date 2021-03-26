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
                left=3,
                right=11,
                top=3,
                bottom=20,
            ),
            CardRect(
                left=-3,
                right=-11,
                top=-3,
                bottom=-20,
            ),
        )
    ),
)

TRUMP_CARDS = CardGroup(
    card_names=frozenset(str(i) for i in range(1, 22)),
    identifiable_rects=frozenset(
        (
            CardRect(
                left=6,
                right=22,
                top=6,
                bottom=23,
            ),
            CardRect(
                left=-6,
                right=-22,
                top=-6,
                bottom=-23,
            ),
        )
    ),
)

FOOL = CardGroup(
    card_names=frozenset("fool"),
    identifiable_rects=frozenset(
        (
            # TODO: This is a guess based on SUITED_CARDS.
            CardRect(
                left=3,
                right=11,
                top=3,
                bottom=11,
            ),
            CardRect(
                left=-3,
                right=-11,
                top=-3,
                bottom=-11,
            ),
        )
    ),
)

TAROT_DECK = Deck(
    width=66,
    height=120,
    cards=frozenset(
        (
            SUITED_CARDS,
            TRUMP_CARDS,
            FOOL,
        )
    ),
)

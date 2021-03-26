from invoke import task, Collection, Task
from card_generator.decks.base import Deck
import importlib


def augment_with_task_decorator(collection: Collection):
    def fn(*args, **kwargs):
        t = task(*args, **kwargs)
        assert isinstance(t, Task)
        collection.add_task(t)
        return t

    collection.task = fn
    return collection


def _get_deck_by_name(deck_module_name: str) -> Deck:
    return importlib.import_module(f"card_generator.decks.{deck_module_name}").DECK

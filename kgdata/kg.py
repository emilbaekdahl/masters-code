from __future__ import annotations

import abc
import concurrent.futures as cf
import functools as ft
import operator
import typing as tp

import pandas as pd

from . import util


class Element(abc.ABC):
    label: str
    kg: KG

    def __init__(self, kg: KG, label: str) -> None:
        self.kg = kg
        self.label = label

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, Element):
            return self.label == other.label

        if isinstance(other, str):
            return self.label == other

        return False

    def __str__(self) -> str:
        return str(self.label)

    def __hash__(self):
        return hash(self.label)


class Entity(Element):
    @ft.lru_cache
    def neighbourhood(self, depth: int = 1) -> SubKG:
        reindexed = self.kg.long_data.reset_index()
        # e_indexed = self.kg.kong_data.set_index("entity")

        idx = ft.reduce(
            lambda acc, _depth: reindexed[
                reindexed["entity"].isin(
                    reindexed.loc[acc][["index"]].merge(reindexed, on="index")["entity"]
                )
            ].index,
            range(depth - 1),
            reindexed[reindexed["entity"] == self].index,
        )

        return SubKG(self.kg, reindexed.loc[idx]["index"].unique())

    @ft.lru_cache
    def _neighbourhood(self, depth: int = 1) -> SubKG:
        indicies = self.kg.long_data[self.kg.long_data["entity"] == self].index.unique()
        sub_kg = SubKG(self.kg, indicies)

        if depth == 1:
            return sub_kg

        nested_neighbourhoods = [
            entity._neighbourhood(depth=depth - 1) for entity in sub_kg.entities
        ]

        return ft.reduce(operator.__or__, nested_neighbourhoods + [sub_kg])


class Relation(Element):
    ...


class ElementContainer(abc.ABC):
    data: tp.Dict[str, Element]

    def __init__(
        self,
        kg: KG,
        labels: tp.Iterable[str] = None,
        klass: tp.Type[Element] = None,
        elements: tp.Iterable[Element] = None,
    ) -> None:
        self.kg = kg
        self.data = {}

        if elements is not None:
            assert len(set(map(operator.attrgetter("__class__"), elements))) == 1

            self.data = {element.label: element for element in elements}

        if labels is not None and klass is not None:
            self.data = {label: klass(self.kg, label) for label in labels}

    def __getitem__(self, index_or_label: tp.Union[str, int]) -> Element:
        if isinstance(index_or_label, int):
            return list(self)[index_or_label]

        return self.data[index_or_label]

    def __setitem(self, label: str, element: Element) -> None:
        self.data[label] = element

    def __iter__(self) -> tp.Iterator[Element]:
        return iter(self.data.values())

    def __contains__(self, other: tp.Any) -> bool:
        if isinstance(other, str):
            return other in self.data

        if isinstance(other, Element):
            return other.label in self.data

        return False

    def __or__(self, other: tp.Any):
        if isinstance(other, self.__class__) and self.kg is other.kg:
            return self.__class__(self.kg, elements=list(self) + list(other))

        raise ValueError()

    def __and__(self, other: tp.Any):
        if isinstance(other, self.__class__) and self.kg is other.kg:
            return self.__class__(self.kg, elements=set(self) & set(other))

        raise ValueError()


class EntityContainer(ElementContainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, klass=Entity, **kwargs)


class RelationContainer(EntityContainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, klass=Relation, **kwargs)


class KG:
    data: pd.DataFrame

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    @util.cached_property
    def long_data(self) -> pd.DataFrame:
        return self.data.melt(
            id_vars="relation", var_name="role", value_name="entity", ignore_index=False
        )

    @util.cached_property
    def entities(self) -> EntityContainer:
        entities = self.long_data["entity"].unique()

        return EntityContainer(self, labels=entities)

    @util.cached_property
    def relations(self) -> RelationContainer:
        relations = self.data["relation"].unique()

        return RelationContainer(self, labels=relations)


class SubKG(KG):
    super_kg: KG
    indicies: pd.Index

    def __init__(self, super_kg: KG, indicies: pd.Index) -> None:
        self.super_kg = super_kg
        self.indicies = indicies

    @util.cached_property
    def data(self):
        return self.super_kg.data.loc[self.indicies]

    @util.cached_property
    def long_data(self):
        return self.super_kg.long_data.loc[self.indicies]

    @util.cached_property
    def entities(self):
        entities = self.long_data["entity"].unique()

        return EntityContainer(
            self.super_kg,
            elements=[self.super_kg.entities[entity] for entity in entities],
        )

    @util.cached_property
    def relations(self):
        relations = self.data["relation"].unique()

        return RelationContainer(
            self.super_kg,
            elements=[self.super_kg.relations[relation] for relation in relations],
        )

    def __or__(self, other):
        if isinstance(other, SubKG):
            return SubKG(self.super_kg, self.indicies.union(other.indicies))

        raise ValueError()

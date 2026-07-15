import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

from utils_future import Log

log = Log('Datum')


@dataclass(frozen=True)
class Query:
    query_str: str

    @cached_property
    def parts(self):
        return self.query_str.split('/')

    @cached_property
    def time_part(self):
        return self.parts[0]

    @cached_property
    def entity_part(self):
        return self.parts[1]

    @cached_property
    def measurement_part(self):
        return self.parts[2]


class Measurement(ABC):
    @classmethod
    def is_match(cls, query_str):
        return query_str == cls.__name__


@dataclass(frozen=True)
class CategoryMeasurement(Measurement):
    label: str

    @classmethod
    @abstractmethod
    def list(cls):
        pass

    @classmethod
    def idx(cls):
        return {m.label: m for m in cls.list()}

    @classmethod
    def from_label(cls, label: str):
        idx = cls.idx()
        if label not in idx:
            raise ValueError(f"Invalid label: {label}")
        return idx[label]

    @classmethod
    def __class_getitem__(cls, label: str):
        return cls.from_label(label)

    def to_json(self):
        return f'{self.__class__.__name__}:{self.label}'


class Religion(CategoryMeasurement):
    @classmethod
    def list(cls):
        return [
            cls("Buddhist"),
            cls("Hindu"),
            cls("Islam"),
            cls("RomanCatholic"),
            cls("OtherChristian"),
            cls("Other"),
        ]


@dataclass(frozen=True)
class Int:
    _value: int

    def __init__(self, value):
        object.__setattr__(self, '_value', int(value))

    def to_json(self):
        return self._value


@dataclass(frozen=True)
class Person(Measurement):
    id: str
    name: str


@dataclass(frozen=True)
class Time:
    _value: str

    def is_match(self, query_str):
        return query_str == self._value


class Sex(CategoryMeasurement):
    @classmethod
    def list(cls):
        return [
            cls("Male"),
            cls("Female"),
        ]


@dataclass(frozen=True)
class Region(CategoryMeasurement):
    id: str
    name: str

    def __init__(self, id: str, name: str):
        object.__setattr__(self, 'label', id)
        object.__setattr__(self, 'id', id)
        object.__setattr__(self, 'name', name)


class District(Region):
    @classmethod
    def list(cls):
        return [
            cls(id="LK-11", name="Colombo"),
            cls(id="LK-12", name="Gampaha"),
        ]


@dataclass(frozen=True)
class Datum:
    time: Time
    entity_class: type[Measurement]
    measurement_idx: dict[str, Measurement]

    def __init__(
        self,
        entity_class: type[Measurement],
        time: Time,
        **measurement_idx: Measurement,
    ):
        assert issubclass(entity_class, Measurement)
        object.__setattr__(self, 'entity_class', entity_class)
        assert isinstance(time, Time)
        object.__setattr__(self, 'time', time)
        object.__setattr__(self, 'measurement_idx', measurement_idx)

    def __hash__(self):
        return hash((self.time, frozenset(self.measurement_idx.items())))

    def is_match_time(self, time_part: str) -> bool:
        if not self.time.is_match(time_part):
            log.warning(f'{time_part} != {self.time}')
            return None
        return self.time._value

    def get_measurement_class_names(self):
        return [type(m).__name__ for m in list(self.measurement_idx.values())]

    def get_non_final_measurement_class_names(self):
        return self.get_measurement_class_names()[:-1]

    def is_match_entity(self, entity_part: str) -> bool:
        if not self.entity_class.__name__ == entity_part:
            log.warning(f'{entity_part} != {self.entity_class.__name__}')
            return None
        return self.entity_class.__name__

    def is_match_measurement_idx(self, measurement_part: str) -> bool:
        class_names_required = measurement_part.split('*')
        matches = {}
        for class_name in class_names_required:
            has_match = False
            for measurement_key, measurement in self.measurement_idx.items():
                if measurement.is_match(class_name):
                    has_match = True
                    matches[class_name] = measurement_key
                    break
            if not has_match:
                return None
        return matches

    def is_match(self, query: Query) -> bool:

        time_part = self.is_match_time(query.time_part)
        entity_part = self.is_match_entity(query.entity_part)
        mesurement_part = self.is_match_measurement_idx(
            query.measurement_part
        )

        if not (time_part and entity_part and mesurement_part):
            return None

        return dict(
            time_part=time_part,
            entity_part=entity_part,
            mesurement_part=mesurement_part,
        )

    def get_measurement_for_class_name(self, class_name):
        for measurement in self.measurement_idx.values():
            if type(measurement).__name__ == class_name:
                return measurement
        return None

    def to_json_inner(self, idx, measurement_part):
        class_names_required = measurement_part.split('*')

        idx_temp = idx
        for i, class_name in enumerate(class_names_required):
            measurement = self.get_measurement_for_class_name(class_name)
            value = measurement.to_json()
            if value not in idx:

                final_value = {}
                if i == len(class_names_required) - 1:
                    for k, v in self.measurement_idx.items():
                        if v.__class__.__name__ in class_names_required:
                            continue
                        final_value[k] = v.to_json()

                idx_temp[value] = final_value

            idx_temp = dict(
                sorted(idx_temp.items(), key=lambda item: item[0])
            )
            idx_temp = idx_temp[value]
        return idx


@dataclass(frozen=True)
class Datumset:
    _value: set[Datum]

    def __init__(self, *data: Datum):
        object.__setattr__(self, '_value', set(data))

    @property
    def first_datum(self):
        return next(iter(self._value))

    def is_match(self, query: Query) -> bool:
        return self.first_datum.is_match(query)


@dataclass(frozen=True)
class MatchedDatumset:
    query: Query
    datumset: Datumset
    match: dict

    def to_json(self):
        idx = {}
        idx_inner = {}
        prev_entity_class = None
        for datum in self.datumset._value:
            entity_class = datum.entity_class.__name__
            if prev_entity_class and prev_entity_class != entity_class:
                raise ValueError(
                    f"Multiple entity classes found:"
                    + f" {prev_entity_class} and {entity_class}"
                )
            time = datum.time._value
            idx_inner = datum.to_json_inner(
                idx_inner, self.query.measurement_part
            )

            if entity_class not in idx:
                idx[entity_class] = {}
            if time not in idx[entity_class]:
                idx[entity_class][time] = []
            idx[entity_class][time] = idx_inner
            prev_entity_class = entity_class

        return dict(
            metadata=self.match,
            data=idx,
        )

    def to_str(self):
        return json.dumps(self.to_json(), indent=4)


class LankaData:
    @classmethod
    def list(cls) -> list[Datumset]:
        return [
            Datumset(
                Datum(
                    Person,
                    Time("2024"),
                    district=District['LK-11'],
                    religion=Religion['Buddhist'],
                    n=Int(100),
                ),
                Datum(
                    Person,
                    Time("2024"),
                    district=District['LK-12'],
                    religion=Religion['Buddhist'],
                    n=Int(200),
                ),
                Datum(
                    Person,
                    Time("2024"),
                    district=District['LK-12'],
                    religion=Religion['Hindu'],
                    n=Int(150),
                ),
            )
        ]

    @classmethod
    def __class_getitem__(cls, query_str):
        query = Query(query_str)
        for datumset in cls.list():
            match = datumset.is_match(query)
            if match:
                return MatchedDatumset(query, datumset, match)

        raise ValueError(f"No matching Datumset found for label: \"{query}\"")


if __name__ == '__main__':

    for query_str in [
        '2024/Person/Religion*District',
        '2024/Person/District*Religion',
        '2024/Person/District',
        '2024/Person/Religion',
    ]:
        print(LankaData[query_str].to_str())
        print('-' * 32)

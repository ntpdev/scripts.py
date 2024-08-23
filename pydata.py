from pydantic import BaseModel, Field, TypeAdapter, StringConstraints, NonNegativeInt, ValidationError
from typing_extensions import Annotated
from typing import List
from uuid import uuid4, UUID
from rich.pretty import pprint
from pathlib import Path


class Person(BaseModel):
    # provide a default value for id, id field is a uuid. pydantic has support for many types from standard library
    # name, age mandatory - use constrained types to validate
    # best_friend optional
    id: UUID = Field(default_factory=uuid4)
    # id: Annotated[str, Field(default_factory=lambda: uuid4().hex)] # type is str
    # see https://docs.pydantic.dev/2.8/api/types/  use Annotated instead of constr
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2)]
    age: NonNegativeInt
    # union types can be written as | or the older way of Union[str, None]
    best_friend: str | None = None


    # can write custom validation but using conint(ge=0) or the builtin types is preferred
    # @field_validator('age')
    # @classmethod
    # def validate_age(cls, v):
    #     if v < 0:
    #         raise ValueError(f'age must be positive')
    #     return v
    

    # provide a custom ordering
    def __lt__(self, other):
        return self.age < other.age


class PersonRepository:
    def __init__(self):
        # create a pydantic model for List[Person] so we can serialise to JSON
        self.adapter = TypeAdapter(List[Person])

    def load(self, fname: Path) -> List[Person]:
        with open(fname, 'r') as f:
            return self.adapter.validate_json(f.read())

    def save(self, xs: List[Person], fname: Path):
        with open(fname, 'wb') as f:
            # dump_json will convert Person to an array of bytes
            # add param indent=2 to make it human readable
            f.write(self.adapter.dump_json(xs, exclude_none=True))


def main():
    person1 = Person(name='John', age=4)  # construct named params
    print(person1)  # default pydantic does not show class name, prints: id='...', name='John' age=4
    data = {'name': 'Janet', 'age': 5, 'best_friend': person1.name}
    person2 = Person(**data)  # construct from dict
    print(person2)  # prints: Person(id='...', name='Jane', age=5)

    person3 = Person.model_validate_json('{"name": "Zed", "age": 1}')  # construct from JSON

    # pydantic models do not validate property sets so person2.age = -9 is allowed
    try:
        person4 = Person(name=" Jake ", age=99)
        pprint(person4)
        person4 = Person(name=" x ", age=-1)
    except ValidationError as e:
        pprint(e)

    xs = [person1, person2, person3]
    # sort by name
    xs.sort(key=lambda e: e.name)
    pprint(xs)
    # sort by default ordering person1 < person2 = True
    xs.sort()
    pprint(xs)

    repo = PersonRepository()
    p = Path.home() / 'Documents' / 'z.json'

    repo.save(xs, p)
    print(f'saved list to {p}')

    ys = repo.load(p)
    print(f'loaded {len(ys)} from {p}')
    pprint(ys)


if __name__ == "__main__":
    main()

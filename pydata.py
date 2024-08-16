from pydantic import BaseModel, field_validator, Field, TypeAdapter, ValidationError
from typing_extensions import Annotated
from typing import List
from uuid import uuid4, UUID
from rich.pretty import pprint
from pathlib import Path

    
class Person(BaseModel):
    # provide a default value for id, id field is a uuid in python but converted to a string
    # name, age mandatory
    # best_friend optional
    # id: str = Field(default_factory=uuid4) # cheating by making the type hint a str but actual type is UUID
    id: Annotated[str, Field(default_factory=lambda: uuid4().hex)] # type is str
    name: str
    age: int
    best_friend: str | None = None

    # use field_validator to convert str to UUID
    # @field_validator('id')
    # @classmethod
    # def validate_id(cls, v):
    #     if len(v) < 16:
    #         raise ValueError(f'uuid must be a string')
    #     return UUID(v)


    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0:
            raise ValueError(f'age must be positive')
        return v
    

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
        person3 = Person(name="Zed", age=-1)
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

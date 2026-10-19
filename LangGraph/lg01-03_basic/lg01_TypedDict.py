from typing import Dict, TypedDict

sample_dict: Dict[str, str] = {
    'name': '김말똥',
    'age': '39',
    'job': '회사원',
}

print(sample_dict)
# ? {'name': '김말똥', 'age': '39', 'job': '회사원'}

# --------------------------------------------------------------------------------


class Person(TypedDict):
    name: str
    age: int
    job: str


person1 = Person(name='김말똥', age=22, job='회사원')
# --------------------------------------------------------------------------------

person2: Person = {
    'name': '김개똥',
    'age': 20,
    'job': '전사',
}

print(person2)
# ? {'name': '김개똥', 'age': 20, 'job': '전사'}

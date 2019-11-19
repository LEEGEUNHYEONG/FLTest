from dataclasses import dataclass


@dataclass(order=True)
class GitData:
    repo : str
    name : str
    html_url : str
    desc :str
    language :str
    forks_count :int
    license :str
    watchers : int
    score : int

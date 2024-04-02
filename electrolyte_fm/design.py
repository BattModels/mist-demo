import pyDOE2
import yaml
import typer
import json
from copy import deepcopy

cli = typer.Typer()


class HyperSpace:
    def __init__(self, data: dict) -> None:
        self._data = data
        self._keys = list(self._data.keys())
        self._keys.sort()

    @property
    def n(self):
        return len(self._keys)

    @property
    def levels(self):
        return [len(self._data[k]) for k in self._keys]

    def get_design(self, experiments: list[list[int]]) -> list[dict]:
        designs = []
        for experiment in experiments:
            designs.append(
                {k: self._data[k][level] for k, level in zip(self._keys, experiment)}
            )
        return designs

    @classmethod
    def from_fid(cls, fid):
        data = yaml.safe_load(fid)
        return HyperSpace(data)


def interpolate_design(config: dict, design: dict):
    """Recursively replace values in confing with entries in design"""
    out = dict()
    for k, v in config.items():
        if isinstance(v, str) and v.startswith("$"):
            out[k] = design[v[1:]]
        elif isinstance(v, dict):
            out[k] = interpolate_design(v, design)
        else:
            out[k] = v

    return out


def design_experiment(f, file: typer.FileText):
    data = yaml.safe_load(file)
    hs = HyperSpace(data["space"])
    target_config = deepcopy(data["config"])
    for expr in hs.get_design(f(hs.levels)):
        config = interpolate_design(target_config, expr)
        config_json = json.dumps(config)
        cmd = data["command"].replace("$json", "--json '" + config_json + "'")
        print(cmd)


@cli.command()
def gsd(file: typer.FileText, r: int = typer.Option(3, "--reduce")):
    def f(levels):
        return pyDOE2.gsd(levels, r)

    design_experiment(f, file)


@cli.command()
def fullfact(file: typer.FileText):
    def f(levels):
        return pyDOE2.fullfact(levels).astype(int).tolist()

    design_experiment(f, file)


if __name__ == "__main__":
    cli()

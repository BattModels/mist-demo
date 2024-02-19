#!/usr/bin/env python
# For Help: submit.py --help
# NOTE: Must activate environment (`poetry shell`) first
from __future__ import annotations
import jinja2
import typer
import os
import yaml
import json
from pathlib import Path


cli = typer.Typer(rich_markup_mode="markdown")


def parse_data(fid, suffix) -> dict:
    if suffix in [".yaml", ".yml"]:
        return yaml.safe_load(fid)
    return json.load(fid)


class RelEnvironment(jinja2.Environment):
    """Override join_path() to enable relative template paths."""

    def join_path(self, template, parent):
        return str(Path(parent).parent.joinpath(template))


@cli.command()
def compose(
    file: str = typer.Argument(
        file_okay=True,
        dir_okay=False,
        help="Jinja2 Template to render",
    ),
    data: str = typer.Option(
        default=None,
        file_okay=True,
        dir_okay=False,
        show_default="file with `*.yaml` suffix",
        help="YAML or JSON file specifying the template's variables",
    ),
):
    """
    Render a template `file` with using the values from `data`

    # Examples

    - Render a Template: `./submit/submit.py submit/polaris.j2`\n
    - Submit on Polaris: `./submit/submit.py submit/polaris.j2 | qsub`\n
    - Use a different config: `./submit/submit.py --data path/to/other/config.yaml submit/polaris.j2`

    """
    env = RelEnvironment(
        loader=jinja2.FileSystemLoader(os.getcwd()),
        lstrip_blocks=False,
    )
    template = env.get_template(file)

    # Load data
    data = data or Path(file).with_suffix(".yaml")
    with open(data, "r") as fid:
        data = parse_data(fid, Path(data).suffix)

    print(template.render(data))


if __name__ == "__main__":
    cli()

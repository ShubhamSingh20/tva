from rich.console import Console

console = Console()


def step(n: int, text: str) -> None:
    console.print(f" [bold green]\u2713 Step {n}[/]  {text}")


def step_error(n: int, text: str) -> None:
    console.print(f" [bold red]\u2717 Step {n}[/]  {text}")

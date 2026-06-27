# Contributing to Mixture of Models

Thanks for your interest in improving MoM! Contributions of all kinds are
welcome — bug reports, documentation, new upstream providers, and engine
improvements.

## Development setup

```bash
git clone https://github.com/luckeyfaraday/MoM-engine.git
cd MoM-engine
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

With no environment variables set, the API runs the deterministic, zero-cost
mock provider, so you can develop and run the full test suite without any API
keys. See [`.env.example`](.env.example) for live-upstream configuration.

## Making changes

1. Fork the repo and create a feature branch off `main`.
2. Keep changes focused; match the surrounding code style.
3. Add or update tests under `tests/` for any behavior change.
4. Make sure `pytest` passes locally — CI runs the same suite on
   Python 3.10, 3.11, and 3.12.
5. Open a pull request describing the change and the motivation.

## Project layout

See the [Project Layout](README.md#project-layout) section of the README for a
map of `mom/api`, `mom/core`, and `mom/providers`.

## Reporting bugs

Open an issue using the bug report template. For security issues, please follow
[SECURITY.md](SECURITY.md) instead of filing a public issue.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).

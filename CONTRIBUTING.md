# Contributing to SimpleChain

First off, thanks for considering contributing! 🎉

## The Golden Rule

**No new dependencies. Ever.**

SimpleChain's entire value proposition is zero dependencies. If your PR adds any external package, it will be rejected. Use Python stdlib only.

## How to Contribute

### Bug Reports

1. Check existing issues first
2. Include Python version
3. Include full error traceback
4. Include minimal reproduction code

### Feature Requests

1. Check if it fits the "simple" philosophy
2. Can it be done without dependencies?
3. Open an issue to discuss first

### Pull Requests

1. Fork the repo
2. Create a branch: `git checkout -b feature/amazing-feature`
3. Make your changes to `simplechain.py`
4. Add tests in `tests/`
5. Run tests: `python -m pytest tests/ -v`
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open a PR

## Code Style

- Follow existing patterns
- Keep functions small and focused
- Add docstrings to public APIs
- Use type hints
- No external formatters required - just keep it clean

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=simplechain

# Run integration tests (needs API keys)
ANTHROPIC_API_KEY=sk-ant-... python -m pytest tests/ -v
```

## What We're Looking For

- Bug fixes
- New provider adapters (Cohere, Together, etc.)
- Documentation improvements
- Performance optimizations
- Better error messages

## What We're NOT Looking For

- New dependencies (seriously, don't)
- Complex abstractions
- "Framework" features
- Breaking changes

## Questions?

Open an issue. We're friendly.

---

Thanks for keeping it simple! 🚀

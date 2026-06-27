# Security Policy

## Supported versions

This project is in active early development. Security fixes are applied to the
`main` branch.

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report privately through GitHub's
[private vulnerability reporting](https://github.com/luckeyfaraday/MoM-engine/security/advisories/new)
(Security tab → "Report a vulnerability"). We aim to acknowledge reports within
a few days and will keep you updated on remediation progress.

## Scope notes

- The OpenCode CLI upstream (`MOM_UPSTREAM=opencode-cli`) shells out to a local
  `opencode` binary and is intended for local development only, not for a
  public hosted deployment.
- Never commit API keys. Configuration is read from environment variables or a
  local, gitignored `.env` file (see [`.env.example`](.env.example)).
- Bearer-token auth for the public surface is available via `MOM_API_KEYS`.

# Security Policy

## Supported versions

The `main` branch is the supported development line. Tagged releases, when available, are supported until they are superseded by a newer release.

## Reporting a vulnerability

Please do not report security vulnerabilities, leaked credentials, or sensitive participant data through public GitHub issues.

Use GitHub's private vulnerability reporting or security advisory flow for this repository when available:

https://github.com/ParvinGhaffarzadeh/multi-sensor-grf-estimation/security/advisories/new

If private reporting is unavailable, contact the project maintainers through a private channel listed by the repository organization or maintainer profile.

## What to include

- Affected version, commit, or branch.
- A minimal reproduction or proof of concept that does not include private participant data.
- Expected impact and affected environment.
- Whether the issue involves package code, notebooks, workflows, dependencies, credentials, datasets, model artifacts, or generated results.

## Scope

Security reports may include vulnerabilities in package code, project automation, dependency handling, unsafe model artifact handling, accidental inclusion of sensitive data, or exposed credentials.

If a secret or private participant artifact has been committed, revoke or rotate the secret if applicable and report the incident privately so maintainers can coordinate remediation.

We aim to acknowledge reports promptly and coordinate a fix before public disclosure when the issue is valid.

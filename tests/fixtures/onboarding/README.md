# Onboarding coverage fixtures

`fastapi-service` and `vite-frontend` contain pinned, partial snapshots of two
independent upstream repositories. `UPSTREAM.json` records commits, source URLs,
original paths, byte counts and hashes. Each includes its upstream MIT license.
The Vite React/TypeScript template is rebased to a fixture root. These subsets
are evidence samples, not runnable checkouts or proof of project-wide coverage.
Do not install their dependencies or execute their code for these tests.

`documented-ui` and `sparse-service` are original, fictional regression fixtures.
They deliberately cover inline commands, a linked guide with a Markdown table,
colocated frontend tests, documented prerequisites/conventions, and absent
practical guidance despite manifest/script/test declarations. Their examples
are not production recommendations. A reviewed explanation is added only to a
temporary copy of the fictional UI during tests; upstream decisions are never
invented or attributed to their maintainers.

Tests copy each fixture into a temporary repository before building knowledge.
No generated maps are checked into these source snapshots. Missing linked docs,
unavailable services, historical command compatibility, full applications and
execution success remain outside their scope.

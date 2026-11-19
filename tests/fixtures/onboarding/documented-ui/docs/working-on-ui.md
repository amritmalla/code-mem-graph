# Working on quote review

## Components

| Path | Responsibility |
| --- | --- |
| `src/pricing/PriceCard.tsx` | Renders the quoted total in integer cents. |
| `src/pricing/PriceCard.spec.tsx` | Contains the PriceCard display regression. |

## Code style

Keep monetary values in integer minor units. The UI displays a server-provided
total; it does not calculate discounts. This keeps pricing policy in one place.

## Verification

From the repository root, run the focused check `pnpm test -- PriceCard.spec.tsx`.
Before proposing any UI change, run both of the following checks:

```sh
pnpm typecheck
pnpm lint
```

These commands are fixture documentation, not evidence of successful execution.

See [the introduction](../README.md) for setup. The reverse link tests cycles.

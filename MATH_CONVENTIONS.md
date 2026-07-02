# Math Formatting Conventions

All course sections follow these rules so formulas render in Markdown viewers **and** read clearly as plain text.

## Display equations (centered, important formulas)

Use double dollar signs on their own lines:

```
$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2
$$
```

**Always follow with a readable line:**

> **Readable form:** predicted y = intercept + (weight₁ × feature₁) + (weight₂ × feature₂)

## Inline math (inside a sentence)

Use single dollar signs: `The slope $w_1$ controls how fast...`

## Currency (US dollars)

Do **not** use bare `$` before numbers in prose — parsers confuse it with math.

| Bad | Good |
|-----|------|
| $18 fare | **18 USD** fare, or fare of 18 dollars |
| $500 fee | **500 USD** project fee |

In Python code strings, `$` inside quotes is fine: `plt.ylabel('Fare (USD)')`

## Symbols in readable form

| Symbol | Say it as |
|--------|-----------|
| $\hat{y}$ | y-hat (predicted value) |
| $\sum$ | sum |
| $\frac{a}{b}$ | a divided by b |
| $w_0$ | w-zero (intercept) |
| $R^2$ | R-squared |
| $\epsilon$ | epsilon |

## Do not use

- `\(...\)` or `\[...\]` — does not render in most Markdown viewers
- Abbreviated formulas without readable explanation

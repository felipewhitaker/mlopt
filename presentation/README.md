# Marp

[Marp](https://marp.app/) is a tool to convert markdown files to slides.

## Usage

```bash
marp --pdf slides.md
```

## Troubleshooting

### Using a custom theme

Update `.vscode/settings.json`

```json
// .vscode/settings.json
{
    "markdown.marp.themes": [
        ".\\presentation\\theme\\border.css",
        // or the raw .css url
        "https://raw.githubusercontent.com/rnd195/my-marp-themes/main/border.css"
    ],
}
```

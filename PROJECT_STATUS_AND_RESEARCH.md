cd "$HOME\Documents\ranked-surfers-ai"

Get-Clipboard -Raw |
  Set-Content ".\PROJECT_STATUS_AND_RESEARCH.md" -Encoding UTF8

Get-Item ".\PROJECT_STATUS_AND_RESEARCH.md" |
  Select-Object Name, Length

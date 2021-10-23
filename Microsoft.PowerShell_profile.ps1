oh-my-posh --init --shell pwsh --config "~/ohmyposhv3-v2.json" | Invoke-Expression

function cd-down { Set-Location ~\Downloads }

function gs { git status }

function glg { git log --graph --decorate --oneline -n 19 }


stty -ixon
export EDITOR=micro
autoload -U edit-command-line
zle -N edit-command-line
bindkey '\ee' edit-command-line
bindkey '\ez' undo
bindkey '\eZ' redo
bindkey '^S' clear-screen
function _cd_back { cd - > /dev/null && zle reset-prompt; }
zle -N _cd_back
bindkey '\ea' _cd_back

alias nv='nvitop -m'
alias c='su - dev -c "cd $(pwd) && claude --dangerously-skip-permissions"'
alias m='micro'
alias me='[[ -f .envrc ]] || echo "dotenv" > .envrc; micro .envrc && direnv allow'
alias mev='[[ -f .envrc ]] || echo -e "dotenv\nexport VIRTUAL_ENV_DISABLE_PROMPT=1\nsource .venv/bin/activate" > .envrc; micro .envrc && direnv allow'
function mec { [[ -f .envrc ]] || echo -e "dotenv\neval \"\$(conda shell.zsh hook)\" && conda activate ${1:-llm}" > .envrc; micro .envrc && direnv allow; }
alias ls='eza --group-directories-first --icons=auto'
alias ll='eza -alh --group-directories-first --git --icons=auto'
alias la='eza -A --group-directories-first --icons=auto'
alias lt='eza --tree --level=2 --icons=auto'
alias ..='cd ..'
alias ...='cd ../..'
alias bat='batcat'
alias fd='fdfind'
alias lg='lazygit'
alias cpa='copypath'
alias cf='copyfile'

eval "$(direnv hook zsh)"
export PATH="$HOME/.local/bin:$PATH"
eval "$(zoxide init zsh)"

setopt INC_APPEND_HISTORY
setopt HIST_IGNORE_DUPS

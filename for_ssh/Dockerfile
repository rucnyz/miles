ARG BASE_IMAGE=radixark/miles:latest
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y docker.io
RUN apt-get update && apt-get install -y openssh-server sudo && \
    mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create dev user only for claude code (can't run --dangerously-skip-permissions as root)
RUN useradd -m -s /bin/bash dev && echo 'dev ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# Install claude code for dev user
USER dev
RUN curl -fsSL https://claude.ai/install.sh | bash
USER root

# SSH key for root
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
COPY for_ssh/id_ed25519.pub /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

# Dev tools (from config/install.sh)
RUN apt-get update && apt-get install -y \
    tmux gpg zsh micro bat chafa fd-find tree tldr duf ncdu direnv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# eza
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO- https://raw.githubusercontent.com/eza-community/eza/main/deb.asc | gpg --dearmor -o /etc/apt/keyrings/gierens.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/gierens.gpg] http://deb.gierens.de stable main" > /etc/apt/sources.list.d/gierens.list && \
    chmod 644 /etc/apt/keyrings/gierens.gpg /etc/apt/sources.list.d/gierens.list && \
    apt-get update && apt-get install -y eza && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# lazygit
RUN LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') && \
    curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz" && \
    tar xf lazygit.tar.gz lazygit && install lazygit /usr/local/bin && rm lazygit lazygit.tar.gz

# Git credentials (from build args)
ARG GH_USER
ARG GH_TOKEN

# Setup for root (main user)
RUN git config --global credential.helper store && \
    echo "https://${GH_USER}:${GH_TOKEN}@github.com" > /root/.git-credentials && \
    git config --global --add safe.directory /root/miles

# tmux for root
WORKDIR /root
RUN git clone https://github.com/rucnyz/.tmux.git && \
    ln -s -f .tmux/.tmux.conf && \
    cp .tmux/.tmux.conf.local .

# zoxide
RUN curl -sSfL https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | sh

# oh-my-zsh + plugins + themes
RUN rm -rf ~/.oh-my-zsh ~/.zshrc && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/.oh-my-zsh/custom/themes/powerlevel10k && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting && \
    git clone --depth 1 https://github.com/Aloxaf/fzf-tab ~/.oh-my-zsh/custom/plugins/fzf-tab

# fzf
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install --all

# p10k config
RUN git clone https://github.com/rucnyz/config.git /tmp/config && \
    cp /tmp/config/.p10k.zsh ~/.p10k.zsh && \
    rm -rf /tmp/config

# fzf preview script
RUN mkdir -p ~/.config/fzf && \
    printf '#!/bin/bash\nmime=$(file -b --mime-type "$1")\nif [[ $mime == image/* ]]; then\n    chafa -s 40x20 "$1"\nelse\n    batcat --color=always "$1" 2>/dev/null || cat "$1"\nfi\n' > ~/.config/fzf/preview.sh && \
    chmod +x ~/.config/fzf/preview.sh

# zsh config
RUN sed -i '/^ZSH_THEME=/c\ZSH_THEME="powerlevel10k/powerlevel10k"' ~/.zshrc && \
    sed -i '/^plugins=(/s/)$/ extract zsh-autosuggestions zsh-syntax-highlighting sudo fzf-tab fzf copypath copyfile)/' ~/.zshrc && \
    sed -i '/\[ -f ~\/.fzf.zsh \]/i\export FZF_DEFAULT_COMMAND="fdfind --type f --hidden --exclude .git"\nexport FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"\nexport FZF_ALT_C_COMMAND="fdfind --type d --hidden --exclude .git"\nexport FZF_DEFAULT_OPTS="--color=light --preview '\''$HOME/.config/fzf/preview.sh {}'\''"\nexport FZF_CTRL_T_OPTS="--preview '\''$HOME/.config/fzf/preview.sh {}'\''"\nexport FZF_ALT_C_OPTS="--preview '\''tree -C {} | head -50'\''"\nexport FZF_CTRL_R_OPTS="--wrap --preview '\'''\''"\n' ~/.zshrc && \
    echo 'source ~/.p10k.zsh' >> ~/.zshrc

# zsh extras (aliases, direnv, zoxide, keybindings)
COPY for_ssh/zshrc_extra.zsh /tmp/zshrc_extra.zsh
RUN cat /tmp/zshrc_extra.zsh >> ~/.zshrc && rm /tmp/zshrc_extra.zsh

# micro config
RUN mkdir -p ~/.config/micro && \
    printf '{\n    "autosu": true,\n    "colorscheme": "simple",\n    "savecursor": true,\n    "saveundo": true,\n    "scrollbar": false,\n    "softwrap": true,\n    "tabsize": 4,\n    "tabstospaces": true\n}\n' > ~/.config/micro/settings.json

# bat config
RUN mkdir -p ~/.config/bat && echo '--theme="Solarized (light)"' > ~/.config/bat/config

# Set root's shell to zsh
RUN chsh -s $(which zsh) root

# Grant dev full access to /root via ACL (works for both existing and future files)
RUN apt-get update && apt-get install -y acl && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN chmod 755 /root && \
    setfacl -R -m u:dev:rwX /root && \
    setfacl -R -d -m u:dev:rwX /root

# OpenSage
ARG GH_TOKEN
RUN git clone https://${GH_TOKEN}@github.com/opensage-agent/opensage-adk-dev.git /root/opensage \
    && pip install --no-cache-dir --ignore-installed -e /root/opensage


EXPOSE 2222

CMD ["/usr/sbin/sshd", "-D", "-p", "2222"]

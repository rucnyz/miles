ARG BASE_IMAGE=slimerl/slime:latest
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y docker.io
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd ~/.ssh && \
    chmod 700 ~/.ssh && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY id_ed25519.pub /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

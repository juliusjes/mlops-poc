pyenv install -s 3.12.10
pyenv virtualenv 3.12.10 env
pyenv local env
pip install -r requirements.txt

# install package
pip install -e ..

# podman
systemctl --user enable --now podman.socket

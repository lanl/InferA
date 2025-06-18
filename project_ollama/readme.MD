# Setting up ollama

1. Installing ollama - follow the first part of these instructions: 
https://copdips.com/2025/03/installing-ollama-without-root.html
```
mkdir -p ~/src
cd ~/src
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
mkdir -p ~/opt/ollama
tar -C ~/opt/ollama -xzf ollama-linux-amd64.tgz
```

2.Run allocate_gpu.sh to get a node with gpu. You should be allocated a node: eg cn111.
```
cd /<your_folder>/project_ollama
source allocate_gpu.sh
```

3. Start ollama: ~/opt/ollama/bin/ollama serve

4. Start a new terminal, connecting to the cluster and then to the node:
```
ssh <your_username>@darwin-fe.lanl.gov
ssh <allocated_node>
```

5. Pull the models you want to use e.g.
```
~/opt/ollama/bin/ollama pull nomic-embed-test:latest
or
~/opt/ollama/bin/ollama pull mistral-small3.1:latest
```

# Create a python environment

1. In your project ollama folder, python environment created via:
```
cd /<your_folder>/project_ollama
python -m venv venv_project_ollama
source venv_project_ollama/bin/activate
```

2. Make sure python environment is activated. Installed modules using:
```
python -m pip install -r requirements.txt
```

# Add genericio module

1. If you haven't setup remote_gio_explorer yet, do that. I copied the genericio directory here to import.


# Running

1. Run allocate_gpu.sh to get a node with gpu
```
source allocate_gpu.sh
```

2. Start ollama: ~/opt/ollama/bin/ollama serve

3. Start a new terminal, connecting to the cluster and then to the node:
```
ssh <your_username>@darwin-fe.lanl.gov
ssh <allocated_node> # e.g ssh cn123
```

4. Load module and activate python environment
```
cd /<your_folder>/project_ollama
source load_modules.sh
source venv_project_ollama/bin/activate
```

5. Run test_ollama.py


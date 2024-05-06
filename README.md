(Optional) Create virtual enviroment:
python3 -m venv [venv_name]

    Activate the virtual enviroment:
    
    source ./venv_name/bin/activate

# Install PorePy from source
    git clone https://github.com/enricoballini/porepy.git

    cd porepy

Switch to `exam` branch:

    git checkout exam

Install requirements:

    pip install -r requirements.txt

Run a simulation:

    cd src/porepy/models/

    python3 two_phase_hu.py

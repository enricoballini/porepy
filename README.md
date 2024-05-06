Create and activate virtual enviroment (Optional):
    
    python3 -m venv venv_name
    
    source ./venv_name/bin/activate

Clone PorePy repository:
    
    git clone https://github.com/enricoballini/porepy.git

    cd porepy

Switch to `exam` branch:

    git checkout exam

Install requirements:

    pip install -r requirements.txt

Run a simulation:

    cd src/porepy/models/

    python3 two_phase_hu.py
